from flask import Flask, render_template, request, send_file, redirect,url_for,redirect
import pandas as pd
from eda import Data_Preprocess
from dtale.app import build_app
from dtale.views import startup
import os
import io
import zipfile
from ydata_profiling import ProfileReport
app = Flask(__name__)
app1 = build_app(reaper_on=False)



name = ''

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/preprocess', methods=['POST'])
def preprocess():
    if request.method == 'POST':
        folder_name = "uploads"

        # List all files in the uploads folder
        files = os.listdir(folder_name)

        # Find the first CSV file
        csv_file = next((file for file in files if file.endswith('.csv')), None)

        if csv_file:
            print("CSV file found:", csv_file)
            file_path = os.path.join(folder_name, csv_file)
            print("File path:", file_path)
        else:
            print("No CSV file found in the 'uploads' folder.")

        df = pd.read_csv(file_path)
        excel_file_name = request.form['excel_file_name']
        if not excel_file_name:
            return "Excel file name not provided"
        global name
        name = excel_file_name
        change_datatype = request.form.get('change_datatype')
        column_name = request.form.get('column_name')
        new_datatype = request.form.get('new_datatype')
        target_column = request.form['target_column']
        data_processor = Data_Preprocess(df, excel_file_name)
        data_processor.run(change_datatype=change_datatype == 'Yes',
                           column_name=column_name,
                           new_datatype=new_datatype,
                           target_column=target_column)
        return render_template('preprocessed.html')
    return "Something went wrong"






# Utility function to read log file
def read_log_file(log_file_path):
    try:
        if not os.path.exists(log_file_path):
            return "Log file not found!"
        with open(log_file_path, 'r') as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Error reading log file: {str(e)}"



@app.route('/logs')
def show_logs():
    global name
    print("name : ",name)
    if not name:
        return redirect('/')
    log_content = read_log_file(f"{name}_logfile.log")
    return render_template('log.html', content=log_content)


 
def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

@app.route('/download')
def download_folder():
    global name
    if not name:
        return "Folder name not provided"
    print("+++*******"+os.getcwd()+"**********++")
    
    folder_path = name  # Using the 'name' variable as the folder name
    folder_path = folder_path.strip('/')
    zip_output_path = f'{folder_path}.zip'  # Define the path for the zip file
    print(f"********{folder_path}********")
    print(f"********{zip_output_path}********")
    print("+++*******"+os.getcwd()+"**********++")
    try:
        zip_folder(folder_path, zip_output_path)  # Create the zip file
        return send_file(f"{name}/"+zip_output_path, as_attachment=True)  # Send the zip file for download
    except FileNotFoundError:
        return "Folder not found", 404




UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = file.filename
        
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('home'))  # Redirect to the home route





@app.route("/show_eda")
def create_df():
    current_directory = os.getcwd()
    print("**************eda****************")
    print(current_directory)
    print(current_directory)
    df = pd.read_csv('eda.csv')
    profile = ProfileReport(df, title="Profiling Report")
    filename = f"{name}_report.html"
    profile.to_file(filename)
    print(current_directory)
    filename = f"{name}/{name}_report.html"
    return send_file(filename)
    
@app.route('/p')
def project_directory():
    # Define the path to your project directory
    project_directory_path = '/'
    # Redirect the user to the project directory
    return redirect(project_directory_path)


if __name__ == "__main__":
    app.run(debug=True)
