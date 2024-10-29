import pickle, os, csv

def save_processed_traj_data(dataset_identifier, traj_data, with_segment=False):
    if with_segment:
        processed_data_path = f"./data/ProcessedData/{dataset_identifier}_with_segment.pkl"
    else:
        processed_data_path = f"./data/ProcessedData/{dataset_identifier}.pkl"
    print(f"Saving processed trajectory data to {processed_data_path}")
    with open(processed_data_path, 'wb') as f:
        pickle.dump(traj_data, f)
    print("Processed trajectory data saved successfully.")

def save_processed_traj_df_data(dataset_identifier, traj_df, with_segment=False):
    if with_segment:
        processed_data_path = f"./data/ProcessedData/{dataset_identifier}_df_with_segment.pkl"
    else:
        processed_data_path = f"./data/ProcessedData/{dataset_identifier}_df.pkl"
    print(f"Saving processed trajectory data to {processed_data_path}")
    with open(processed_data_path, 'wb') as f:
        pickle.dump(traj_df, f)
    print("Processed trajectory data saved successfully.")

def save_processed_traj_bounds_data(dataset_identifier, bounds, with_segment=False):
    if with_segment:
        processed_data_path = f"./data/ProcessedData/{dataset_identifier}_bounds_with_segment.pkl"
    else:
        processed_data_path = f"./data/ProcessedData/{dataset_identifier}_bounds.pkl"
    print(f"Saving processed trajectory data to {processed_data_path}")
    with open(processed_data_path, 'wb') as f:
        pickle.dump(bounds, f)
    print("Processed trajectory data saved successfully.")

def load_processed_traj_data(dataset_identifier, with_segment=False):
    if with_segment:
        processed_data_path = f"./data/ProcessedData/{dataset_identifier}_with_segment.pkl"
        df_processed_data_path = f"./data/ProcessedData/{dataset_identifier}_df_with_segment.pkl"
        bounds_processed_data_path = f"./data/ProcessedData/{dataset_identifier}_bounds_with_segment.pkl"
    else:
        processed_data_path = f"./data/ProcessedData/{dataset_identifier}.pkl"
        df_processed_data_path = f"./data/ProcessedData/{dataset_identifier}_df.pkl"
        bounds_processed_data_path = f"./data/ProcessedData/{dataset_identifier}_bounds.pkl"
    print(f"Loading processed trajectory data from {processed_data_path}")
    with open(processed_data_path, 'rb') as f:
        traj_data = pickle.load(f)
    with open(df_processed_data_path, 'rb') as f:
        traj_df = pickle.load(f)
    with open(bounds_processed_data_path, 'rb') as f:
        bounds = pickle.load(f)
    
    return traj_df, bounds, traj_data, dataset_identifier

def save_index(index_identifier, index_data):
    index_path = f"./data/Index/{index_identifier}.pkl"
    with open(index_path, 'wb') as f:
        pickle.dump(index_data, f)
    print(f"Index saved to {index_path}")

def load_index(index_identifier):
    index_path = f"./data/Index/{index_identifier}.pkl"
    with open(index_path, 'rb') as f:
        return pickle.load(f)

def save_index_statistics(index_identifier, build_time, build_memory):
    output_csv = "./result/IndexStatistics.csv"
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["Index Identifier", "Build Time (s)", "Build Memory (MB)"])
        csv_writer.writerow([index_identifier, build_time, build_memory])

def save_dataset_statistics(dataset_identifier, output_csv, data_size_mb, trajectory_number, recordNum, avg_time_interval, selected_interval):
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["Dataset Identifier", "Data Size (Mb)", "Trajectory Number", "Record Number", "Avg Record Number per Trajectory", "Avg Time Interval", "Selected Interval"])
        csv_writer.writerow([dataset_identifier, f"{data_size_mb:.2f}", trajectory_number, recordNum, f"{recordNum / trajectory_number:.2f}",  f"{avg_time_interval:.2f}", f"{selected_interval:.2f}"])

