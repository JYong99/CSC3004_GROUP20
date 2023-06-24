from flask import Flask, render_template, request, redirect, url_for, session
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import numpy as np
import time
from PIL import Image
import torch
from torchvision import models, transforms
import io

#Generate a secret key
secret_key = os.urandom(24)

#Initialize Flask
app = Flask(__name__)
app.secret_key = secret_key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

#Connect to MongoDB
client = MongoClient('mongodb://host.docker.internal:27017')
db = client['cloudProj']
users = db['myCollection']

#Optional: Insert a new user for demonstration
#Usually, you would add users through a registration process
password_hash = generate_password_hash('password123')
new_user = {'username': 'csc3004', 'password': password_hash}
users.insert_one(new_user)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=False)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = torch.nn.Linear(model.fc.in_features, 25) # 25 classes
model.load_state_dict(torch.load('sign_language_resnet50.pth', map_location=device))
model = model.to(device)
model.eval()

# Prepare the transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Check if the given filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Render the login page
@app.route('/')
def login_page():
    return render_template('index.html')

# Handle the login process
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    # Check if the username and password match in the database
    users = db.users
    user = users.find_one({'username': username})

    if user and check_password_hash(user['password'], password):
        session['username'] = username
        return redirect(url_for('upload'))
    else:
        return redirect(url_for('login_page'))
    
# Render the image upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # Check if the user is logged in
    if 'username' not in session:
        return redirect(url_for('login_page'))

    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']

        # Check if a file is selected and has an allowed extension
        if file and allowed_file(file.filename):
            # Read the file contents
            file_contents = file.read()

            # Predict the class of the uploaded image
            prediction = get_prediction(image_bytes=file_contents)

            return render_template('upload.html', prediction=prediction)
        else:
            return render_template('upload.html', prediction='Invalid file!')

    return render_template('upload.html')

# Mapping from class_id to actual sign language letters
id_to_char = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'K',
    10: 'L',
    11: 'M',
    12: 'N',
    13: 'O',
    14: 'P',
    15: 'Q',
    16: 'R',
    17: 'S',
    18: 'T',
    19: 'U',
    20: 'V',
    21: 'W',
    22: 'X',
    23: 'Y'
    # No 'Z'
}

# Modify the get_prediction function to return the actual letter
def get_prediction(image_bytes):
    def transform_image(image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        return transform(np.array(image)).unsqueeze(0)
    
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor.to(device))
    _, y_hat = outputs.max(1)
    return id_to_char[y_hat.item()]

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
