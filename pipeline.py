from steps.ingestion import ingest_data

if __name__ == "__main__":
    data = ingest_data()
    print(data.head())