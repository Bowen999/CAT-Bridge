import tkinter as tk
from tkinter import filedialog, messagebox, StringVar, OptionMenu, ttk
from PIL import Image, ImageTk
import subprocess
import shutil

class MyApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.geometry("650x600")  # Increased height for new elements
        self.title('CAT Bridge')
        self.create_widgets()

    def create_widgets(self):
        # Load image file
        img = Image.open("img/logo.png")
        img = img.resize((200, 150), Image.ANTIALIAS)  # Resize to required size
        self.logo = ImageTk.PhotoImage(img)

        # Add label to display image
        self.logo_label = tk.Label(self, image=self.logo)
        self.logo_label.place(x=250, y=20)

        labels = ['Transcriptome', 'Metabolome', 'Study Design (optional)', 'Gene Annotation (optional)', 'Target', 'Cluster Count', 'Method', 'OpenAI API Key (optional)']
        for i, label in enumerate(labels):
            tk.Label(self, text=label).place(x=50, y=200 + i * 30)

        self.gene_file_button = tk.Button(self, text="Choose File", command=lambda: self.load_file(self.gene_file_button, 'gene_file'))
        self.gene_file_button.place(x=350, y=200)

        self.meta_file_button = tk.Button(self, text="Choose File", command=lambda: self.load_file(self.meta_file_button, 'metabo_file'))
        self.meta_file_button.place(x=350, y=230)

        self.design_file_button = tk.Button(self, text="Choose File", command=lambda: self.load_file(self.design_file_button, 'design_file'))
        self.design_file_button.place(x=350, y=260)

        self.annotation_file_button = tk.Button(self, text="Choose File", command=lambda: self.load_file(self.annotation_file_button, 'annotation_file'))
        self.annotation_file_button.place(x=350, y=290)

        self.target_entry = tk.Entry(self)
        self.target_entry.place(x=350, y=320)

        self.count_variable = StringVar(self)
        self.count_variable.set("8")  # Default value
        self.count_menu = OptionMenu(self, self.count_variable, *[str(i) for i in range(1, 11)])
        self.count_menu.place(x=350, y=350)

        self.function_variable = StringVar(self)
        self.function_variable.set("CCM (groups>4)")  # Default value
        self.function_menu = OptionMenu(self, self.function_variable, "CCM (groups>4)", "Granger (groups>4)", "CCA (group>4)", "DWT (group>2)", "CCF (group>2)", "Spearman (group>2)", "Pearson (group>2)")
        self.function_menu.place(x=350, y=380)

        self.api_key_entry = tk.Entry(self)
        self.api_key_entry.place(x=350, y=410)

        # Define style
        style = ttk.Style()
        style.configure("TButton", font=('Helvetica', 12, 'bold'), background='blue', foreground='black')

        self.submit_button = ttk.Button(self, text="Submit", command=self.submit, style="TButton")
        self.submit_button.place(x=220, y=470)

        self.reset_button = ttk.Button(self, text="Reset", command=self.reset, style="TButton")
        self.reset_button.place(x=320, y=470)

    def load_file(self, button, variable_name):
        filename = filedialog.askopenfilename()
        if filename:
            button.configure(text=filename)
            setattr(self, variable_name, filename)

    def submit(self):
        gene_file = self.gene_file_button['text'] or "no"
        metabo_file = self.meta_file_button['text'] or "no"
        design_file = self.design_file_button['text'] if self.design_file_button['text'] != "Choose File" else "no"
        annotation_file = self.annotation_file_button['text'] if self.annotation_file_button['text'] != "Choose File" else "no"
        target = self.target_entry.get()
        cluster_count = self.count_variable.get()
        f = self.function_variable.get()
        api_key = self.api_key_entry.get() if self.api_key_entry.get() else None 

        # Check if all required files and fields are filled
        if not gene_file or not metabo_file or not target:
            messagebox.showerror("Error", "Transcriptome, Metabolome, and Target are required!")
            return  # Exit function, do not execute the code below

        # If Study Design or Gene Annotation is empty, use "no"
        # if not design_file:
        #     design_file = "no"
        # if not annotation_file:
        #     annotation_file = "no"

        # # Run the script
        # subprocess.run(["python", "run.py", gene_file, metabo_file, design_file, annotation_file, target, cluster_count, f, api_key])
        # args = ["python", "run.py", gene_file, metabo_file, design_file or "no", annotation_file or "no", target, cluster_count, f]
       
        python_executable = shutil.which("python") or shutil.which("python3.9")
        if not python_executable:
            raise RuntimeError("Neither 'python' nor 'python3.9' is available on the system.")
        args = [python_executable, "run.py", gene_file, metabo_file, design_file or "no", annotation_file or "no", target, cluster_count, f]

        if api_key:
            args.append(api_key)
        
        try:
            subprocess.run(args, check=True)
            messagebox.showinfo("Success", "Task completed successfully!")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"An error occurred while running the script: {e}")
        
    def reset(self):
        # Reset button
        self.gene_file_button.configure(text="Choose File")
        self.meta_file_button.configure(text="Choose File")
        self.design_file_button.configure(text="Choose File")
        self.annotation_file_button.configure(text="Choose File")
        self.target_entry.delete(0, tk.END)
        self.count_variable.set("8")
        self.function_variable.set("CCM")
        self.api_key_entry.delete(0, tk.END)

if __name__ == "__main__":
    app = MyApp()
    app.mainloop()
