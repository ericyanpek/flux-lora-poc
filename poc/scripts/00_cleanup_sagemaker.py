"""
One-time cleanup of SageMaker POC leftovers.
- Removes non-latest ECR images (saves ~$0.10/GB/month storage)
- Removes flux-poc-s3-access inline policy from SageMaker role
Run: python3 00_cleanup_sagemaker.py
"""
import boto3

ACCOUNT = "984072314535"
REGION = "us-east-1"
ECR_REPO = "flux-poc-training"
SM_ROLE_NAME = "AmazonSageMaker-ExecutionRole-20250207T115166"


def cleanup_ecr_old_images():
    ecr = boto3.client("ecr", region_name=REGION)
    paginator = ecr.get_paginator("describe_images")
    images = [img for page in paginator.paginate(repositoryName=ECR_REPO) for img in page["imageDetails"]]

    # find the digest tagged as 'latest'
    latest_digest = None
    for img in images:
        if "latest" in img.get("imageTags", []):
            latest_digest = img["imageDigest"]
            break

    to_delete = [
        {"imageDigest": img["imageDigest"]}
        for img in images
        if img["imageDigest"] != latest_digest
    ]

    if not to_delete:
        print("ECR: no old images to delete")
        return

    ecr.batch_delete_image(repositoryName=ECR_REPO, imageIds=to_delete)
    print(f"ECR: deleted {len(to_delete)} old image(s), kept latest ({latest_digest[:19]}...)")


def cleanup_sm_iam_policy():
    iam = boto3.client("iam", region_name=REGION)
    policy_name = "flux-poc-s3-access"
    try:
        iam.delete_role_policy(RoleName=SM_ROLE_NAME, PolicyName=policy_name)
        print(f"IAM: removed inline policy '{policy_name}' from {SM_ROLE_NAME}")
    except iam.exceptions.NoSuchEntityException:
        print(f"IAM: policy '{policy_name}' not found (already removed or never applied)")


if __name__ == "__main__":
    cleanup_ecr_old_images()
    cleanup_sm_iam_policy()
    print("\n✅ SageMaker cleanup complete")
