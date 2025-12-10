from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import mysql.connector
from deep_translator import GoogleTranslator
import re
import requests
from PIL import Image
import pytesseract
import os
from urllib.parse import urlparse

app = Flask(__name__)
app.secret_key = '1c8073775dbc85a92ce20ebd44fd6a4fd832078f59ef16ec'


ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

nltk.download('punkt')
nltk.download('stopwords')


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():  # keep only alphanumeric
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))  # stemming

    return " ".join(y)


# --Detecting a spam link ----------------------
def is_spam_link(url):
    url_pattern = re.compile(
        r'^(https?:\/\/)?'                 
        r'(([A-Za-z0-9-]+\.)+[A-Za-z]{2,})'
        r'(\/[^\s]*)?$'
    )
    if not url_pattern.match(url):
        return True

    parsed = urlparse(url)
    domain = parsed.netloc.lower()


    bad_domains = ['.xyz', '.top', '.ru', '.cn', '.zip', '.tk']
    if any(domain.endswith(ext) for ext in bad_domains):
        return True

    #Keywords that can not be ignored and are treated as suspicious
    bad_keywords = ['free', 'login', 'gift', 'claim', 'verify', 'secure', 'update', 'win', 'lottery']
    if any(word in domain or word in parsed.path.lower() for word in bad_keywords):
        return True


    try:
        response = requests.head(url, timeout=3)
        if response.status_code >= 400:
            return True
    except Exception:
        return True

    return False

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="123",
    database="smc"
)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/index')
def index():
    if 'user' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('signin'))

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get('message')

    if not input_text or input_text.strip() == "":
        return render_template('result.html', prediction="Please enter a message or link!")

    #  Checking if the link is spam or not:
    if re.match(r'http[s]?://', input_text.strip()):
        if is_spam_link(input_text.strip()):
            prediction = " Spam Link Detected"
        else:
            prediction = "Link is safe"
        return render_template('result.html', prediction=prediction)


    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(input_text)
    except Exception:
        translated_text = input_text

    transformed_text = transform_text(translated_text)
    vector_input = tfidf.transform([transformed_text])
    result = model.predict(vector_input)[0]

    prediction = "Spam Message" if result == 1 else "Not Spam"
    return render_template('result.html', prediction=prediction)


# Image detection  using OCR. Not working
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('result.html', prediction=" Please upload an image")

    image_file = request.files['image']
    if image_file.filename == '':
        return render_template('result.html', prediction=" No file selected")

    image_path = os.path.join("static", "uploaded_image.jpg")
    image_file.save(image_path)

    extracted_text = pytesseract.image_to_string(Image.open(image_path))

    transformed = transform_text(extracted_text)
    vector = tfidf.transform([transformed])
    result = model.predict(vector)[0]


    if result == 1:
        prediction = "Spam Image Email Detected"
    else:
        prediction = "Not A Spam Image"

    return render_template('result.html', prediction=prediction)


@app.route('/signin')
def signin():
    if 'user' in session:
        return redirect(url_for('index'))
    return render_template('signin.html')


@app.route('/signup', methods=['GET'])
def signup():
    return render_template('signup.html')


@app.route('/register', methods=['POST'])
def register():
    full_name = request.form['full_name']
    username = request.form['username']
    email = request.form['email']
    phone = request.form['phone']
    password = request.form['password']
    confirm_password = request.form['confirm_password']

    if password != confirm_password:
        flash('Password and Confirm Password do not match.', 'danger')
        return redirect('/signup')

    try:
        cur = db.cursor()
        cur.execute("""
            INSERT INTO users (full_name, username, email, phone, password)
            VALUES (%s, %s, %s, %s, %s)
        """, (full_name, username, email, phone, password))
        db.commit()
        cur.close()
        flash('Registration successful! Please log in.', 'success')
        return redirect('/signin')
    except Exception as e:
        flash(f'Error: {e}', 'danger')
        return redirect('/signup')


@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']
    remember_me = request.form.get('remember_me')

    cur = db.cursor()
    cur.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
    user = cur.fetchone()
    cur.close()

    if user:
        session['user'] = user
        if remember_me:
            session.permanent = True
        return redirect(url_for('index'))
    else:
        flash('Invalid email or password!', 'danger')
        return redirect('/signin')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully!', 'info')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

