import tkinter as tk
from tkinter import filedialog, messagebox, StringVar, OptionMenu, ttk
from PIL import Image, ImageTk
import subprocess
import shutil
import os

class MyApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.geometry("650x600")  
        self.title('CAT Bridge')

        # Dictionary to store file paths
        self.file_paths = {}

        # Initialize selected method
        self.selected_method = "CCM"

        # Create all widgets
        self.create_widgets()

    def load_file(self, button, variable_name):
        """Handles file selection and updates button text."""
        filename = filedialog.askopenfilename()
        if filename:
            self.file_paths[variable_name] = filename  
            button.configure(text=os.path.basename(filename))  
        else:
            self.file_paths[variable_name] = None  

    def submit(self):
        """Handles form submission and runs `run.py` with selected parameters."""
        gene_file = self.file_paths.get('gene_file', "no")
        metabo_file = self.file_paths.get('metabo_file', "no")
        design_file = self.file_paths.get('design_file', "no")
        annotation_file = self.file_paths.get('annotation_file', "no")
        target = self.target_entry.get()
        cluster_count = self.count_variable.get()
        method_clean = self.selected_method  
        api_key = self.api_key_entry.get() if self.api_key_entry.get() else None

        if gene_file == "no" or metabo_file == "no" or not target:
            messagebox.showerror("Error", "Transcriptome, Metabolome, and Target are required!")
            return  

        python_executable = shutil.which("python3") or shutil.which("python") or shutil.which("python3.9")
        if not python_executable:
            raise RuntimeError("Python is not available on the system.")

        args = [python_executable, "run.py", gene_file, metabo_file, design_file, annotation_file, target, cluster_count, method_clean]

        if api_key:
            args.append(api_key)

        try:
            print(f"Running command: {' '.join(args)}")  
            subprocess.run(args, check=True)  
            messagebox.showinfo("Success", "Task completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")  
            messagebox.showerror("Error", f"An error occurred: {e}")

    def reset(self):
        """Resets all form fields to their default values."""
        self.file_paths.clear()  
        self.gene_file_button.configure(text="Choose File")
        self.meta_file_button.configure(text="Choose File")
        self.design_file_button.configure(text="Choose File")
        self.annotation_file_button.configure(text="Choose File")
        self.target_entry.delete(0, tk.END)
        self.count_variable.set("8")
        self.function_variable.set(list(self.method_mapping.keys())[0])  
        self.selected_method = list(self.method_mapping.values())[0]  
        self.api_key_entry.delete(0, tk.END)

    def create_widgets(self):
        """Creates UI elements after functions are defined."""
        img = Image.open("img/logo.png")
        img = img.resize((200, 150), Image.LANCZOS)  
        self.logo = ImageTk.PhotoImage(img)

        self.logo_label = tk.Label(self, image=self.logo)
        self.logo_label.place(x=250, y=20)

        labels = [
            'Transcriptome', 'Metabolome', 'Study Design (optional)',
            'Gene Annotation (optional)', 'Target', 'Cluster Count',
            'Method', 'OpenAI API Key (optional)'
        ]

        for i, label in enumerate(labels):
            tk.Label(self, text=label).place(x=50, y=200 + i * 30)

        # File selection buttons
        self.gene_file_button = tk.Button(self, text="Choose File", command=lambda: self.load_file(self.gene_file_button, 'gene_file'))
        self.gene_file_button.place(x=350, y=200)

        self.meta_file_button = tk.Button(self, text="Choose File", command=lambda: self.load_file(self.meta_file_button, 'metabo_file'))
        self.meta_file_button.place(x=350, y=230)

        self.design_file_button = tk.Button(self, text="Choose File", command=lambda: self.load_file(self.design_file_button, 'design_file'))
        self.design_file_button.place(x=350, y=260)

        self.annotation_file_button = tk.Button(self, text="Choose File", command=lambda: self.load_file(self.annotation_file_button, 'annotation_file'))
        self.annotation_file_button.place(x=350, y=290)

        # Input fields
        self.target_entry = tk.Entry(self)
        self.target_entry.place(x=350, y=320)

        self.count_variable = StringVar(self)
        self.count_variable.set("8")  
        self.count_menu = OptionMenu(self, self.count_variable, *[str(i) for i in range(1, 11)])
        self.count_menu.place(x=350, y=350)

        # Method Mapping (Now without parentheses!)
        self.method_mapping = {
            "CCM (groups>4)": "CCM",
            "Granger (groups>4)": "Granger",
            "CCA (group>4)": "CCA",
            "DTW (group>2)": "DTW",
            "CCF (group>2)": "CCF",
            "Spearman (group>2)": "Spearman",
            "Pearson (group>2)": "Pearson"
        }

        method_display_values = list(self.method_mapping.keys())  

        self.function_variable = StringVar(self)
        self.function_variable.set(method_display_values[0])  

        def update_method_selection(choice):
            """Updates selected method with cleaned name (without parentheses)."""
            self.selected_method = self.method_mapping.get(choice, "CCM")

        self.function_menu = OptionMenu(self, self.function_variable, *method_display_values, command=update_method_selection)
        self.function_menu.place(x=350, y=380)

        self.api_key_entry = tk.Entry(self)
        self.api_key_entry.place(x=350, y=410)

        # Define button styles
        style = ttk.Style()
        style.configure("TButton", font=('Helvetica', 12, 'bold'))

        self.submit_button = ttk.Button(self, text="Submit", command=self.submit, style="TButton")
        self.submit_button.place(x=220, y=470)

        self.reset_button = ttk.Button(self, text="Reset", command=self.reset, style="TButton")
        self.reset_button.place(x=320, y=470)

if __name__ == "__main__":
    app = MyApp()
    app.mainloop()
