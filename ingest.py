from rag.ingestion.ingestion_engine import ingest_folder

data_folder = "data"

print("Starting ingestion...\n")

ingest_folder(data_folder)

print("\nIngestion completed successfully.")