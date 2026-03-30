import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from dotenv import load_dotenv
from urllib.parse import urlparse
import gc
import requests
load_dotenv()
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, ImageOps
from flask import Flask, render_template, request, redirect, session, send_from_directory, url_for, flash, jsonify
import smtplib
from email.mime.text import MIMEText
import random, pickle
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
import imagehash
from itsdangerous import URLSafeTimedSerializer
import cv2
import cloudinary
import cloudinary.uploader

import psycopg2

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def highlight_similarity(img1_path, img2_path):

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        return None
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(500)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 5:
        return None

    pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)

    x,y,w,h = cv2.boundingRect(pts)

    # Draw rectangle
    result = img1.copy()
    cv2.rectangle(result, (x,y), (x+w,y+h), (0,0,255), 3)

    output_path = f"highlight_{random.randint(1000,9999)}.jpg"
    cv2.imwrite(output_path, result)

    return output_path

def get_db_connection():
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        print("DATABASE_URL missing")
        return None

    # Fix for postgres:// issue (safe)
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    return psycopg2.connect(db_url)

cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("API_SECRET")
)

# ---------------- CONFIG ----------------
SIMILARITY_THRESHOLD = 0.80
ADMIN_EMAIL = "24h51a05r1@cmrcet.ac.in"

print("Loading AI model...")

model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    alpha=0.35
)

print("AI Model Loaded Successfully")

def get_model():
    return model

# ---------------- UTILITY FUNCTIONS ----------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))
def get_image_hash(img_path):

    img = Image.open(img_path)

    # Fix rotation
    img = ImageOps.exif_transpose(img)

    # Generate perceptual hash
    phash = imagehash.phash(img)

    return str(phash)


def get_embedding(img_path):

    # Validate image
    try:
        img_check = Image.open(img_path)
        img_check.verify()
    except Exception:
        raise ValueError("Invalid image file")

    # Open image
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img = img.resize((128,128))  # smaller = less memory

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    model = get_model()
    embedding = model(img_array, training=False).numpy()

    return embedding.flatten()

def orb_similarity(img1_path, img2_path):

    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)

    img1 = cv2.resize(img1, (500,500))
    img2 = cv2.resize(img2, (500,500))

    if img1 is None or img2 is None:
        return 0

    orb = cv2.ORB_create(1000)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    similarity = len(matches) / max(len(kp1), len(kp2))
    return similarity

def detect_logo_inside(upload_path, existing_path):

    img1 = cv2.imread(upload_path, 0)
    img2 = cv2.imread(existing_path, 0)
    img1 = cv2.resize(img1, (500,500))
    img2 = cv2.resize(img2, (500,500))

    if img1 is None or img2 is None:
        return False

    orb = cv2.ORB_create(1000)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    matches = bf.knnMatch(des1, des2, k=2)

    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) > 30:
        return True

    return False 


def send_email(to_email, subject, message):
    sender_email = os.getenv("EMAIL_USER")
    sender_password = os.getenv("EMAIL_PASS")

    if not sender_email or not sender_password:
        print("Email config missing")
        return

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            
            msg = MIMEText(message)
            msg["Subject"] = subject
            msg["From"] = sender_email
            msg["To"] = to_email

            server.sendmail(sender_email, to_email, msg.as_string())

    except Exception as e:
        print("Email failed:", e)
# ---------------- APP ----------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24))

serializer = URLSafeTimedSerializer(app.secret_key)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jfif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- ROUTES ----------------


@app.route("/")
def home():
    conn = get_db_connection()
    if conn is None:
        return "Database connection failed"

    cursor = conn.cursor()

    return render_template("login.html")


@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            error = "Email already registered. Please login."
        else:
            cursor.execute(
                "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                (name, email, password)
            )
            conn.commit()
            conn.close()
            flash("Registration successful. Please login.", "success")
            return redirect(url_for('home'))

        conn.close()

    return render_template("register.html", error=error)

@app.route("/login", methods=["GET", "POST"])
def login():

    # ---------------- GET REQUEST ----------------
    if request.method == "GET":
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT email FROM users")
        emails = [row[0] for row in cursor.fetchall()]

        conn.close()

        return render_template("login.html", emails=emails)

    # ---------------- POST REQUEST ----------------
    email = request.form.get("email")
    password = request.form.get("password")

    # 🔐 Admin login (hardcoded)
    if email == "similarityinference.ai@gmail.com" and password == "similarityinference.ai":
        session["user_email"] = email
        session["user_name"] = "Admin"
        session["role"] = "admin"
        return redirect("/admin")

    # ---------------- USER LOGIN ----------------
    conn = get_db_connection()
    if conn is None:
        flash("Database connection failed", "error")
        return redirect("/login")

    cursor = conn.cursor()

    try:
        # Get user by email only
        cursor.execute(
            "SELECT id, name, email, password FROM users WHERE email=%s",
            (email,)
        )
        user = cursor.fetchone()

        print("Fetched user:", user)  # 🔍 Debug log

        # ✅ Check password properly
        if user and check_password_hash(user[3], password):
            session["user_email"] = user[2]
            session["user_name"] = user[1]
            session["role"] = "user"

            print("Login success")  # 🔍 Debug
            return redirect("/dashboard")

        else:
            print("Login failed")  # 🔍 Debug
            flash("Invalid email or password", "error")
            return redirect("/login")

    except Exception as e:
        print("Login error:", e)
        flash("Something went wrong", "error")
        return redirect("/login")

    finally:
        conn.close()

@app.route("/dashboard")
def dashboard():

    if "user_email" not in session:
        return redirect("/")

    conn = get_db_connection()
    cursor = conn.cursor()

    if session.get("role") == "admin":
        cursor.execute("SELECT image_url, user_email, score, label FROM uploads")
        images = cursor.fetchall()
        is_admin = True
    else:
        cursor.execute(
            "SELECT image_url, score, label FROM uploads WHERE user_email=%s",
            (session["user_email"],)
        )
        images = cursor.fetchall()
        is_admin = False

    conn.close()

    return render_template(
        "dashboard.html",
        name=session.get("user_name"),
        email=session.get("user_email"),
        images=images,
        is_admin=is_admin
    )

@app.route('/file/<filename>')
def view_file(filename):
    user_folder = os.path.join("uploads", session["user_email"])
    return send_from_directory(user_folder, filename)

@app.route('/view/<filename>')
def preview_image(filename):
    if "user_email" not in session:
        return redirect("/")

    return render_template("view_image.html", filename=filename)

from urllib.parse import unquote

@app.route("/delete/<path:image_url>", methods=["POST"])
def delete_file(image_url):

    if "user_email" not in session:
        return jsonify({"status": "error", "message": "Not logged in"})

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT user_email FROM uploads WHERE image_url=%s", (image_url,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return jsonify({"status": "error", "message": "File not found"})

    # 🔥 DELETE FROM CLOUDINARY
    try:
        path = urlparse(image_url).path
        public_id = os.path.splitext(os.path.basename(path))[0]
        cloudinary.uploader.destroy(public_id)
    except Exception as e:
        print("Cloudinary delete failed:", e)

    # DELETE FROM DB
    cursor.execute("DELETE FROM uploads WHERE image_url=%s", (image_url,))
    conn.commit()
    conn.close()

    return jsonify({"status": "success"})


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect("/")


@app.route("/admin")
def admin_panel():
    # 🔐 Allow only admin
    if session.get("role") != "admin":
        return redirect(url_for("dashboard"))

    conn = get_db_connection()
    cur = conn.cursor()

    # 👥 Get all users
    cur.execute("SELECT id, name, email FROM users")
    users = cur.fetchall()

    # 🖼 Get all uploaded images
    cur.execute("SELECT image_url, user_email, score, label FROM uploads")
    images = cur.fetchall()

    # 📊 Statistics
    cur.execute("SELECT COUNT(*) FROM users")
    total_users = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM uploads")
    total_images = cur.fetchone()[0]

    conn.close()

    return render_template(
        "admin.html",
        users=users,
        images=images,
        total_users=total_users,
        total_images=total_images
    )

@app.route("/admin/delete/<filename>", methods=["POST"])
def admin_delete_image(filename):
    if session.get("role") != "admin":
        return redirect(url_for("dashboard"))

    conn = get_db_connection()
    cursor = conn.cursor()

    # Get file owner
    cursor.execute("SELECT user_email FROM uploads WHERE filename=%s", (filename,))
    row = cursor.fetchone()

    if row:
        user_email = row[0]
        filepath = os.path.join("uploads", user_email, filename)

        if os.path.exists(filepath):
            os.remove(filepath)

        cursor.execute("DELETE FROM uploads WHERE filename=%s", (filename,))
        conn.commit()

    conn.close()
    flash("Image deleted by admin", "success")
    return redirect(url_for("admin_panel"))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():

    if request.method == "POST":

        email = request.form.get("email")

        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT email FROM users WHERE email=%s", (email,))
            user = cursor.fetchone()

            if not user:
                flash("Email not registered")
                return redirect("/forgot_password")

            token = generate_reset_token(email)

            reset_url = url_for(
                'reset_password',
                token=token,
                _external=True
            )

            send_email(
                email,
                "Password Reset",
                f"Click this link to reset your password:\n\n{reset_url}"
            )

            flash("Password reset link sent to your email.")
            return redirect("/login")

        finally:
            conn.close()

    return render_template("forgot_password.html")

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):

    email = verify_reset_token(token)

    if not email:
        flash("Invalid or expired token")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":

        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        # ✅ Check passwords match
        if password != confirm_password:
            flash("Passwords do not match")
            return redirect(request.url)

        # ✅ Hash the password
        hashed_password = generate_password_hash(password)

        # ✅ Update in database
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE users SET password=%s WHERE email=%s",
            (hashed_password, email)
        )

        conn.commit()
        conn.close()

        flash("Password reset successful. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("reset_password.html")

def generate_reset_token(email):
    return serializer.dumps(email, salt='password-reset-salt')


def verify_reset_token(token, expiration=3600):
    try:
        email = serializer.loads(
            token,
            salt='password-reset-salt',
            max_age=expiration
        )
    except:
        return None
    return email


@app.route("/upload", methods=["GET", "POST"])
def upload_file():

    if session.get("role") == "admin":
        return redirect(url_for("dashboard"))

    if "user_email" not in session:
        return redirect(url_for("home"))

    if request.method == "GET":
        return render_template("upload.html")

    file = request.files.get("image")
    label = request.form.get("label", "").strip()

    if label == "":
        return jsonify({"status": "error", "message": "Image title is required"})

    if not file or file.filename == "":
        return jsonify({"status": "error", "message": "No file selected."})

    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Only PNG, JPG, JPEG allowed"})

    # ---------------- TEMP FILE ----------------
    temp_path = f"temp_{random.randint(1000,9999)}.jpg"
    file.save(temp_path)

    # ---------------- AI PROCESS ----------------
    try:
        new_embedding = get_embedding(temp_path)
        new_hash = get_image_hash(temp_path)
    except Exception as e:
        print("UPLOAD ERROR:", e)
        os.remove(temp_path)
        return jsonify({"status": "error", "message": str(e)})

    embedding_blob = pickle.dumps(new_embedding)

    conn = get_db_connection()
    if conn is None:
        return jsonify({"status": "error", "message": "Database connection failed"})

    cursor = conn.cursor()

    cursor.execute("""
        SELECT image_url, user_email, embedding, image_hash, label 
        FROM uploads 
        ORDER BY id DESC 
        LIMIT 10
    """)
    all_uploads = cursor.fetchall()

    highest_score = 0
    matched_filename = None
    matched_user = None
    highlight_url = None
    orb_score = 0
    best_highlight = None

    for url, email, emb, img_hash, lbl in all_uploads:

        existing_temp = None

        try:
            response = requests.get(url, timeout=10, stream=True)
            existing_temp = f"temp_existing_{random.randint(1000,9999)}.jpg"

            with open(existing_temp, "wb") as f:
                f.write(response.content)

            existing_hash = imagehash.hex_to_hash(img_hash)
            new_hash_obj = imagehash.hex_to_hash(new_hash)

            distance = new_hash_obj - existing_hash
            phash_score = 1 - (distance / 64)

            # 🔥 ADD THIS LINE
            orb_score = orb_similarity(temp_path, existing_temp)

            # 🔥 FINAL COMBINED SCORE
            combined_score = (phash_score * 0.6) + (orb_score * 0.4)

            highlight_path = highlight_similarity(temp_path, existing_temp)

            current_highlight = None   # 🔥 use temp variable

            if highlight_path:
                try:
                    upload_res = cloudinary.uploader.upload(highlight_path)
                    current_highlight = upload_res["secure_url"]
                    os.remove(highlight_path)
                except:
                    current_highlight = None

            if combined_score > highest_score:
                highest_score = combined_score
                matched_filename = url
                matched_user = email
                best_highlight = current_highlight   # 🔥 SAVE BEST ONE

        except Exception as e:
            print("Error:", e)

        finally:
            if existing_temp and os.path.exists(existing_temp):
                os.remove(existing_temp)
    gc.collect()

    similarity_score = int(highest_score * 100)

    if highest_score == 0:
        similarity_status = "unique"
    elif similarity_score < 60:
        similarity_status = "unique"
    elif similarity_score < 75:
        similarity_status = "borderline"
    else:
        similarity_status = "duplicate"

    # ---------------- DUPLICATE ----------------
    if similarity_status == "duplicate" and matched_user != session["user_email"]:

        similarity_score = int(highest_score * 100)   # ✅ FORCE THIS

        try:
            send_email(
                matched_user,
                "⚠ Copyright Alert",
                f"Your image is {similarity_score}% similar."
            )

            send_email(
                session["user_email"],
                "⚠ Copyright Warning",
                f"Your upload is {similarity_score}% similar."
            )
        except:
            pass

        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        print("ORB SCORE:", orb_score)
        print("FINAL SCORE:", similarity_score)

        return jsonify({
            "status": "duplicate",
            "score": similarity_score,   # ✅ NOW ALWAYS DEFINED
            "filename": matched_filename,
            "owner": matched_user,
            "highlight": best_highlight   # 🔥 IMPORTANT
        })


    # ✅ UPLOAD for unique + borderline
    try:
        upload_result = cloudinary.uploader.upload(temp_path)
        image_url = upload_result["secure_url"]
    except Exception as e:
        print("CLOUDINARY ERROR:", e)
        return jsonify({"status": "error", "message": "Cloud upload failed"})

    # delete temp file
    os.remove(temp_path)

    # ---------------- SAVE ----------------
    cursor.execute(
        "INSERT INTO uploads (image_url, user_email, score, embedding, image_hash, label) VALUES (%s, %s, %s, %s, %s, %s)",
        (image_url, session["user_email"], similarity_score, embedding_blob, new_hash, label)
    )

    conn.commit()
    cursor.close()
    conn.close()
    gc.collect()

    return jsonify({
        "status": similarity_status,   # unique / borderline / duplicate
        "score": similarity_score,
        "message": "Image uploaded successfully"
    })




@app.route("/init_db")
def init_db():
    conn = get_db_connection()
    if conn is None:
        return "DB connection failed"

    cur = conn.cursor()

    # USERS TABLE
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)

    # UPLOADS TABLE
    cur.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id SERIAL PRIMARY KEY,
            image_url TEXT,
            user_email TEXT,
            score INT,
            embedding BYTEA,
            image_hash TEXT,
            label TEXT
        )
    """)

    conn.commit()
    conn.close()

    return "Tables created successfully!"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
