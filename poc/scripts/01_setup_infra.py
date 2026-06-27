"""
One-time infrastructure setup for EC2-based FLUX.2-dev training.
Creates: S3 bucket, ECR repo, EC2 IAM role + instance profile.
Run: python3 01_setup_infra.py
"""
import boto3
import json

ACCOUNT = "984072314535"
REGION = "us-east-1"
BUCKET = f"flux-poc-{ACCOUNT}-{REGION}"
ECR_REPO = "flux-poc-training"
ROLE_NAME = "flux-poc-ec2-role"
PROFILE_NAME = "flux-poc-ec2-instance-profile"


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

    # 1. Create role with EC2 trust policy
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

    # 2. Attach SSM managed policy
    iam.attach_role_policy(
        RoleName=ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
    )

    # 3. Inline policy: S3 + ECR
    inline = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::{BUCKET}",
                    f"arn:aws:s3:::{BUCKET}/*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                ],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": ["ssm:GetParameter"],
                "Resource": "arn:aws:ssm:us-east-1:984072314535:parameter/flux-poc/*",
            },
        ],
    })
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="flux-poc-ec2-policy", PolicyDocument=inline)
    print("IAM: S3 + ECR inline policy applied")

    # 4. Create instance profile and add role
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


if __name__ == "__main__":
    create_bucket()
    create_ecr_repo()
    create_ec2_iam_profile()
    print(f"\n✅ Infrastructure ready")
    print(f"  S3:      s3://{BUCKET}/")
    print(f"  ECR:     {ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO}")
    print(f"  Profile: {PROFILE_NAME}")
