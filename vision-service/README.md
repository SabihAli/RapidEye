# vision-service

Real-time computer-vision microservice for RapidEye. Receives encoded video from ingestion-service over gRPC, decodes locally, and runs the CV pipeline. See [PLAN.md](./PLAN.md) for the full design. Ingestion contract: [ingestion-service/PLAN.md](../ingestion-service/PLAN.md).
