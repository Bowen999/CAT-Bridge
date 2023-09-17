from django.shortcuts import HttpResponse, render

#def hello(request):
#    return HttpResponse("Hello World!")

def home(request):
    return render(request,"home.html")

def cat(request):
    return render(request, "cat.html")
    # return render(request,"cat_result.html", {"ticks": "1688616186.1213353"})


def about(request):
    return render(request, "about.html")

def tutorial(request):
    return render(request, 'tutorial.html')

#def cat_result(request):
#    return render(request, "cat_result.html", {"ticks": "1688616186.1213353"})

def contact(request):
    return render(request, "contact.html")

def cat_result(request):
    import time
    import os
    import pandas

    ticks = str(time.time())
    os.mkdir('myapp/result/' + ticks)
    if request.method == 'POST':
        print(request.FILES)

        # Check if the file_source is 0 or 1
        file_source = request.POST.get('file_source')
        if file_source == '0':
            # User uploaded files, process them
            a = request.FILES.get("gene_file")
            if a:
                with open('myapp/result/{}/'.format(ticks) + "gene_file.tsv", "wb") as f:
                    for line in a:
                        f.write(line)
            else:
                print(1)
                return render(request, "cat.html")

            a = request.FILES.get("meta_file")
            if a:
                with open('myapp/result/{}/'.format(ticks) + "meta_file.tsv", "wb") as f:
                    for line in a:
                        f.write(line)
            else:
                return render(request, "cat.html")

            a = request.FILES.get("design_file")
            if a:
                with open('myapp/result/{}/'.format(ticks) + "design_file.tsv", "wb") as f:
                    for line in a:
                        f.write(line)
            else:
                return render(request, "cat.html")

            annotation = 0
            a = request.FILES.get("annotation_file")
            if a:
                with open('myapp/result/{}/'.format(ticks) + "annotation_file.tsv", "wb") as f:
                    for line in a:
                        f.write(line)
                annotation = 1
            else:
                annotation = "no"

        else:
            # Use default files from the specified paths
            # Assuming you have these files in the 'pepper' folder
            pepper_dir = os.path.join(os.path.dirname(__file__), "pepper")
            default_files = [
                os.path.join(pepper_dir, "count.tsv"),
                os.path.join(pepper_dir, "metabo.tsv"),
                os.path.join(pepper_dir, "design.tsv"),
                os.path.join(pepper_dir, "gene_annotation.tsv"),
            ]
            filename = ["gene_file.tsv", "meta_file.tsv", "design_file.tsv", "annotation_file.tsv"]
            for i, file_path in enumerate(default_files):
                print("Copying file:", file_path)
                with open(file_path, "rb") as file:
                    content = file.read()
                    dest_path = os.path.join('myapp/result/{}/'.format(ticks), filename[i])
                    print("Destination path:", dest_path)
                    with open(dest_path, "wb") as f:
                        f.write(content)
                
            annotation = 1  # Assuming annotation is always present in the default files

        target = request.POST['target']
        print(target)
        if len(target) == 0:
            return render(request, "cat.html")

        cluster_count = request.POST.get('count', None)
        function = request.POST.get('function', None)
        print(cluster_count)
        print(function)

        os.system("dos2unix " + 'myapp/result/{}/'.format(ticks) + "gene_file.tsv")
        os.system("dos2unix " + 'myapp/result/{}/'.format(ticks) + "meta_file.tsv")
        os.system("dos2unix " + 'myapp/result/{}/'.format(ticks) +
                  "design_file.tsv")
        os.system("dos2unix " + 'myapp/result/{}/'.format(ticks) +
                  "annotation_file.tsv")

        if annotation == 1:
            print("python3 myapp/run.py " + 'myapp/result/{}/'.format(ticks) + "gene_file.tsv " + 'myapp/result/{}/'.format(ticks) + "meta_file.tsv " + 'myapp/result/{}/'.format(
                ticks) + "design_file.tsv " + 'myapp/result/{}/'.format(ticks) + "annotation_file.tsv " + "\"" + target + "\"" + " " + cluster_count + " " + function + " " + ticks)
            os.system("python3 myapp/run.py " + 'myapp/result/{}/'.format(ticks) + "gene_file.tsv " + 'myapp/result/{}/'.format(ticks) + "meta_file.tsv " + 'myapp/result/{}/'.format(
                ticks) + "design_file.tsv " + 'myapp/result/{}/'.format(ticks) + "annotation_file.tsv " + "\"" + target + "\"" + " " + cluster_count + " " + function + " " + ticks)

        elif annotation == "no":
            print("python3 myapp/run.py " + 'myapp/result/{}/'.format(ticks) + "gene_file.tsv " + 'myapp/result/{}/'.format(ticks) + "meta_file.tsv " + 'myapp/result/{}/'.format(
                ticks) + "design_file.tsv " + 'myapp/result/{}/'.format(ticks) + "annotation_file.tsv " + "\"" + target + "\"" + " " + cluster_count + " " + function + " " + ticks)
            os.system("python3 myapp/run.py " + 'myapp/result/{}/'.format(ticks) + "gene_file.tsv " + 'myapp/result/{}/'.format(ticks) + "meta_file.tsv " +
                      'myapp/result/{}/'.format(ticks) + "design_file.tsv " + "no " + "\"" + target + "\"" + " " + cluster_count + " " + function + " " + ticks)
        os.mkdir("myapp/static/" + ticks)
        names = os.listdir("myapp/result/{}/".format(ticks) + "plot/")
        for i in names:
            os.system("cp " + "myapp/result/{}/".format(ticks) +
                      "plot/" + i + " " + "myapp/static/img/" + ticks + "_" + i)

    return render(request, "cat_result.html", {"ticks": ticks})

