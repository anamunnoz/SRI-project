import tkinter as tk
from tkinter import messagebox
from pipeline import run_pipeline

# Global variables to store n, and metrics
valor_n = 0
all_metrics = []

def save_values():
    """
    Retrieve and validate user input for the quantity (n). Show an error message if invalid.
    Update the global variable `valor_n` with the valid input.
    """
    global valor_n
    
    try:
        # Get the values entered by the user
        valor_n = int(entry_cantidad.get())
        
        if valor_n <= 0:
            messagebox.showerror("Error", "The quantity must be a positive number.")
            return
        
        # Show a success message
        messagebox.showinfo("Success", f"Value saved\n Wait for metrics")
    
    except ValueError:
        messagebox.showerror("Error", "Please enter valid values.")
        
    metrics=run_pipeline(valor_n)

#def update(metrics):
    """
    Update the text area with the given metrics. Each metric is preceded by a descriptive text.
    
    Parameters:
        metrics (list of float): A list of five floating-point numbers representing various metrics.
    """
    global all_metrics
    
    # Save the received values
    all_metrics = metrics
    
    # Clear the text area
    resultado_text.delete(1.0, tk.END)
    
    # Text to show before each floating-point number
    prefix_text = ["Precision: ", "Recall: ", "HR: ", "MRR: ", "nDCG: "]
    
    # Display metrics with the preceding text
    resultado_text.insert(tk.END, "\nMetrics:\n")
    for i, num in enumerate(all_metrics):
        resultado_text.insert(tk.END, f"{prefix_text[i]}{num:.4f}\n")

# Create the main window
ventana = tk.Tk()
ventana.title("Recommendation")

# Label and entry for the quantity
tk.Label(ventana, text="Truncation for metrics (n):").grid(row=1, column=0, padx=10, pady=10)
entry_cantidad = tk.Entry(ventana)
entry_cantidad.grid(row=1, column=1, padx=10, pady=10)

# Button to save the value
boton_guardar = tk.Button(ventana, text="Save Value", command=save_values)
boton_guardar.grid(row=2, columnspan=2, padx=10, pady=10)

# Text area to display the metrics
resultado_text = tk.Text(ventana, height=15, width=40)
resultado_text.grid(row=3, columnspan=2, padx=10, pady=10)


ventana.mainloop()
