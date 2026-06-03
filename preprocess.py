import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_DIR = "."
OUTPUT_DIR = "dataset_final_split"  # پوشه خروجی نهایی
INPUT_LEN = 400
# نسبت‌های تقسیم‌بندی
VAL_SIZE = 0.2   # 20% برای Validation از Part I

# تنظیمات برای TL
USE_PART2_FOR_TL = True  # از Part II برای TL استفاده می‌کند
TL_TRAIN_SIZE_PART2 = 0.8  # درصدی از Part II که برای TL آموزش استفاده می‌شود

# فایل‌های داده
FILES = {
    "part1": {
        "x": "DB_THREE_P1_x_SIGNALS.csv",
        "y": "DB_THREE_P1_y_SIGNALS.csv",
        "z": "DB_THREE_P1_z_SIGNALS.csv",
        "labels": "DB_THREE_P1_LOCATONS.csv"
    },
    "part2": {
        "x": "DB_THREE_P2_x_SIGNALS.csv",
        "y": "DB_THREE_P2_y_SIGNALS.csv",
        "z": "DB_THREE_P2_z_SIGNALS.csv",
        "labels": "DB_THREE_P2_LOCATONS.csv"
    }
}


# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_interleaved_gzip(filepath):
    print(f"Loading: {filepath}")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath, compression="gzip",
                         header=0, low_memory=False)
        data_cols = [c for c in df.columns if str(c).isdigit()]
        if not data_cols:
            data_cols = df.columns[1:]

        df_data = df[data_cols]
        signals_only = df_data.iloc[::2, :]
        signals_np = signals_only.T.to_numpy(dtype=np.float32)
        return signals_np
    except Exception as e:
        print(f"Error while loading {filepath}: {e}")
        return None


def load_labels_safe(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)
        return df.values
    except Exception:
        return pd.read_csv(filepath, header=None).values


def crop_and_pad(signal, center_idx, length):
    half_len = length // 2
    start = center_idx - half_len
    end = center_idx + half_len

    pad_l = 0
    if start < 0:
        pad_l = -start
        start = 0

    crop = signal[start:end]

    pad_r = length - len(crop) - pad_l
    if pad_r < 0:
        pad_r = 0

    if pad_l > 0 or pad_r > 0:
        crop = np.pad(crop, (pad_l, pad_r), mode="constant")

    return crop[:length]


def normalize(arr):
    mx = np.max(np.abs(arr))
    return arr / mx if mx > 1e-9 else arr


def preprocess_three_axis_dataset(files_dict, input_len=400):
    """
    خروجی:
      X_all: shape = (N, 3, input_len)
      Y_all: shape = (N, ...)
      raw_example: برای رسم نمونه
    """
    sig_x = load_interleaved_gzip(files_dict["x"])
    sig_y = load_interleaved_gzip(files_dict["y"])
    sig_z = load_interleaved_gzip(files_dict["z"])
    labels = load_labels_safe(files_dict["labels"])

    if sig_x is None or sig_y is None or sig_z is None or labels is None:
        raise FileNotFoundError(
            "One or more dataset files could not be loaded.")

    n_samples = min(len(sig_x), len(sig_y), len(sig_z), len(labels))
    print(f"Common sample count: {n_samples}")

    sig_x = sig_x[:n_samples]
    sig_y = sig_y[:n_samples]
    sig_z = sig_z[:n_samples]
    labels = labels[:n_samples]

    X_processed = []

    for i in range(n_samples):
        rx = np.nan_to_num(sig_x[i], nan=0.0, posinf=0.0, neginf=0.0)
        ry = np.nan_to_num(sig_y[i], nan=0.0, posinf=0.0, neginf=0.0)
        rz = np.nan_to_num(sig_z[i], nan=0.0, posinf=0.0, neginf=0.0)

        # پیدا کردن نقطه همزمان‌سازی از بین سه سنسور
        px = np.argmax(np.abs(rx))
        py = np.argmax(np.abs(ry))
        pz = np.argmax(np.abs(rz))

        peaks = [px, py, pz]
        amps = [np.abs(rx[px]), np.abs(ry[py]), np.abs(rz[pz])]
        ref_idx = peaks[np.argmax(amps)]

        cx = normalize(crop_and_pad(rx, ref_idx, input_len))
        cy = normalize(crop_and_pad(ry, ref_idx, input_len))
        cz = normalize(crop_and_pad(rz, ref_idx, input_len))

        # shape: (3, input_len)
        stacked_sample = np.stack([cx, cy, cz], axis=0)
        X_processed.append(stacked_sample)

    X_all = np.array(X_processed, dtype=np.float32)
    Y_all = np.array(labels)

    raw_example = {
        "sig_x": sig_x,
        "sig_y": sig_y,
        "sig_z": sig_z
    }

    print(f"Processed shape: X={X_all.shape}, Y={Y_all.shape}")
    return X_all, Y_all, raw_example


def save_split(output_dir, prefix, X, y):
    np.save(os.path.join(output_dir, f"X_{prefix}.npy"), X)
    np.save(os.path.join(output_dir, f"y_{prefix}.npy"), y)
    print(f"Saved: X_{prefix}.npy | y_{prefix}.npy")


def plot_raw_example(raw_example, title, save_path, sample_idx=0):
    plt.figure(figsize=(10, 6))
    plt.plot(raw_example["sig_x"][sample_idx], label="Sensor X", alpha=0.8)
    plt.plot(raw_example["sig_y"][sample_idx], label="Sensor Y", alpha=0.8)
    plt.plot(raw_example["sig_z"][sample_idx], label="Sensor Z", alpha=0.8)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved: {save_path}")


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)

    # -----------------------------
    # 1) Preprocess Part I
    # -----------------------------
    print("\n====================")
    print("Processing Part I ...")
    print("====================")
    X_part1, y_part1, raw_part1 = preprocess_three_axis_dataset(
        FILES["part1"],
        input_len=INPUT_LEN
    )

    # Part I -> base train / val
    X_train, X_val, y_train, y_val = train_test_split(
        X_part1,
        y_part1,
        test_size=VAL_SIZE,
        random_state=42,
        shuffle=True
    )

    print("\nPart I split:")
    print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}   | y_val:   {y_val.shape}")

    save_split(OUTPUT_DIR, "train", X_train, y_train)
    save_split(OUTPUT_DIR, "val", X_val, y_val)

    plot_raw_example(
        raw_part1,
        title="Part I - Raw Example Before/For Preprocessing Reference",
        save_path=os.path.join(OUTPUT_DIR, "part1_reference_plot.png"),
        sample_idx=0
    )

    # -----------------------------
    # 2) Preprocess Part II
    # -----------------------------
    print("\n====================")
    print("Processing Part II ...")
    print("====================")
    X_part2, y_part2, raw_part2 = preprocess_three_axis_dataset(
        FILES["part2"],
        input_len=INPUT_LEN
    )

    plot_raw_example(
        raw_part2,
        title="Part II - Raw Example Before/For Preprocessing Reference",
        save_path=os.path.join(OUTPUT_DIR, "part2_reference_plot.png"),
        sample_idx=0
    )

    # -----------------------------
    # 3) Split Part II for TL / Test
    # -----------------------------
    if USE_PART2_FOR_TL:
        X_tl_train, X_test, y_tl_train, y_test = train_test_split(
            X_part2,
            y_part2,
            train_size=TL_TRAIN_SIZE_PART2,
            random_state=42,
            shuffle=True
        )

        print("\nPart II split for TL:")
        print(
            f"  X_tl_train: {X_tl_train.shape} | y_tl_train: {y_tl_train.shape}")
        print(f"  X_test:     {X_test.shape}     | y_test:     {y_test.shape}")

        save_split(OUTPUT_DIR, "tl_train", X_tl_train, y_tl_train)
        save_split(OUTPUT_DIR, "test", X_test, y_test)

    print("\nAll preprocessing and splitting completed successfully.")
    print(f"Output directory: {OUTPUT_DIR}")

    print("\nGenerated files:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {fname}")


if __name__ == "__main__":
    main()
