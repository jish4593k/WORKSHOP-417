import torch
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_csv_and_process(file_path):
   
    df = pd.read_csv(file_path)

   
    data = df.select_dtypes(include=['number']).values

   
    tensor_data = torch.tensor(data, dtype=torch.float32)

   
    result = torch.sum(tensor_data, dim=0)

    return result

def plot_tensor_heatmap(tensor):
   
    sns.heatmap(tensor.view(1, -1).numpy(), annot=True, cmap="YlGnBu", cbar=False)
    plt.title("Seaborn Heatmap of PyTorch Tensor")
    plt.show()

def browse_file_path():
    
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        result_tensor = load_csv_and_process(file_path)
        plot_tensor_heatmap(result_tensor)


root = tk.Tk()
root.title("PyTorch, Tkinter, and Seaborn Example")


browse_button = ttk.Button(root, text="Browse CSV File", command=browse_file_path)
browse_button.pack(pady=20)


root.mainloop()
