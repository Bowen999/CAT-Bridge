import tkinter as tk
from tkinter import filedialog, StringVar, OptionMenu, ttk
from PIL import Image, ImageTk
import subprocess

class MyApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.geometry("650x530")  # set window size
        self.title('CAT Bridge')
        self.create_widgets()

    def create_widgets(self):
        # Load the image file
        img = Image.open("img/logo.png")
        img = img.resize((200, 150), Image.ANTIALIAS)  # Resize to desired size
        self.logo = ImageTk.PhotoImage(img)

        # Add a label to display the image
        self.logo_label = tk.Label(self, image=self.logo)
        self.logo_label.place(x=250, y=20)

        labels = ['Transcriptomics', 'Metabolomics', 'Study Design (optional)', 'Gene Annotation(optional)', 'Target', 'Cluster Count', 'Aggregation Function']
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
        self.count_variable.set("8")  # default value
        self.count_menu = OptionMenu(self, self.count_variable, *[str(i) for i in range(1, 11)])
        self.count_menu.place(x=350, y=350)

        self.function_variable = StringVar(self)
        self.function_variable.set("CCM")  # default value
        self.function_menu = OptionMenu(self, self.function_variable, "CCM", "Granger", "CCA", "DWT", "CCF", "Spearman", "Pearson")
        self.function_menu.place(x=350, y=380)

        # Define the style
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
        gene_file = self.gene_file_button['text']
        metabo_file = self.meta_file_button['text']
        design_file = self.design_file_button['text']
        annotation_file = self.annotation_file_button['text']
        target = self.target_entry.get()
        cluster_count = self.count_variable.get()
        f = self.function_variable.get()

        # Run the script with the specified variables
        subprocess.run(["python", "run.py", gene_file, metabo_file, design_file, annotation_file, target, cluster_count, f])
        # print("python", "run.py", gene_file, metabo_file, design_file, annotation_file, target, cluster_count, f)

    def reset(self):
        # Add your reset code here
        pass

if __name__ == "__main__":
    app = MyApp()
    app.mainloop()
