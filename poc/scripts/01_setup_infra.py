"""
One-time infrastructure setup for EC2-based FLUX.2-dev training.
Creates: S3 bucket, ECR repo, EC2 IAM role + instance profile.
Run: python3 01_setup_infra.py
"""
import boto3
import json
import time
from config import (
    ACCOUNT, REGION, BUCKET, ECR_REPO, ROLE_NAME, PROFILE_NAME,
    CODEBUILD_PROJECT, CODEBUILD_ROLE, CW_LOG_GROUP,
)


def create_bucket():
    s3 = boto3.client("s3", region_name=REGION)
    try:
        s3.create_bucket(Bucket=BUCKET)
        print(f"S3: created bucket {BUCKET}")
    except Exception as e:
        if "BucketAlreadyOwnedByYou" in str(e) or "BucketAlreadyExists" in str(e):
            print(f"S3: bucket already exists")
        else:
            raise
    s3.put_public_access_block(
        Bucket=BUCKET,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
        },
    )
    for prefix in ["datasets/", "outputs/", "checkpoints/"]:
        s3.put_object(Bucket=BUCKET, Key=prefix)
    print("S3: folder structure OK")


def create_ecr_repo():
    ecr = boto3.client("ecr", region_name=REGION)
    try:
        ecr.create_repository(
            repositoryName=ECR_REPO,
            imageScanningConfiguration={"scanOnPush": True},
        )
        print(f"ECR: created repo {ECR_REPO}")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        print(f"ECR: repo already exists")


def create_ec2_iam_profile():
    iam = boto3.client("iam", region_name=REGION)

    trust = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })
    try:
        iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=trust)
        print(f"IAM: created role {ROLE_NAME}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"IAM: role already exists")

    iam.attach_role_policy(
        RoleName=ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
    )

    inline = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{BUCKET}", f"arn:aws:s3:::{BUCKET}/*"],
            },
            {
                "Effect": "Allow",
                "Action": ["ecr:GetAuthorizationToken", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": ["ssm:GetParameter"],
                "Resource": f"arn:aws:ssm:{REGION}:{ACCOUNT}:parameter/flux-poc/*",
            },
        ],
    })
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="flux-poc-ec2-policy", PolicyDocument=inline)
    print("IAM: S3 + ECR + SSM inline policy applied")

    try:
        iam.create_instance_profile(InstanceProfileName=PROFILE_NAME)
        print(f"IAM: created instance profile {PROFILE_NAME}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"IAM: instance profile already exists")

    try:
        iam.add_role_to_instance_profile(InstanceProfileName=PROFILE_NAME, RoleName=ROLE_NAME)
        print(f"IAM: role attached to instance profile")
    except iam.exceptions.LimitExceededException:
        print(f"IAM: role already attached to instance profile")


def create_codebuild_project():
    iam = boto3.client("iam", region_name=REGION)
    cb = boto3.client("codebuild", region_name=REGION)

    # 1. IAM role for CodeBuild
    trust = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "codebuild.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })
    try:
        iam.create_role(RoleName=CODEBUILD_ROLE, AssumeRolePolicyDocument=trust)
        print(f"IAM: created role {CODEBUILD_ROLE}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"IAM: role {CODEBUILD_ROLE} already exists")

    inline = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:InitiateLayerUpload",
                    "ecr:UploadLayerPart",
                    "ecr:CompleteLayerUpload",
                    "ecr:PutImage",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                ],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": [
                    f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:{CW_LOG_GROUP}",
                    f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:{CW_LOG_GROUP}:log-stream:*",
                ],
            },
        ],
    })
    iam.put_role_policy(
        RoleName=CODEBUILD_ROLE,
        PolicyName="flux-poc-codebuild-policy",
        PolicyDocument=inline,
    )
    print(f"IAM: CodeBuild policy applied to {CODEBUILD_ROLE}")

    # 2. CodeBuild project (retry for IAM role propagation delay)
    role_arn = f"arn:aws:iam::{ACCOUNT}:role/{CODEBUILD_ROLE}"
    for attempt in range(6):
        try:
            cb.create_project(
            name=CODEBUILD_PROJECT,
            source={
                "type": "GITHUB",
                "location": "https://github.com/ericyanpek/flux-lora-poc",
                "buildspec": "poc/buildspec.yml",
            },
            environment={
                "type": "LINUX_CONTAINER",
                "image": "aws/codebuild/standard:7.0",
                "computeType": "BUILD_GENERAL1_LARGE",
                "privilegedMode": True,
                "environmentVariables": [
                    {"name": "AWS_ACCOUNT", "value": ACCOUNT, "type": "PLAINTEXT"},
                    {"name": "AWS_REGION", "value": REGION, "type": "PLAINTEXT"},
                ],
            },
            artifacts={"type": "NO_ARTIFACTS"},
            serviceRole=role_arn,
            logsConfig={
                "cloudWatchLogs": {
                    "status": "ENABLED",
                    "groupName": CW_LOG_GROUP,
                    "streamName": "build",
                },
            },
            )
            print(f"CodeBuild: created project {CODEBUILD_PROJECT}")
            break
        except cb.exceptions.ResourceAlreadyExistsException:
            print(f"CodeBuild: project {CODEBUILD_PROJECT} already exists")
            break
        except Exception as e:
            if "InvalidInputException" in str(e) and attempt < 5:
                print(f"IAM role not yet propagated, retrying in 10s... (attempt {attempt+1}/6)")
                time.sleep(10)
            else:
                raise


if __name__ == "__main__":
    create_bucket()
    create_ecr_repo()
    create_ec2_iam_profile()
    create_codebuild_project()
    print(f"\n✅ Infrastructure ready")
    print(f"  S3:        s3://{BUCKET}/")
    print(f"  ECR:       {ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO}")
    print(f"  Profile:   {PROFILE_NAME}")
    print(f"  CodeBuild: {CODEBUILD_PROJECT}")
