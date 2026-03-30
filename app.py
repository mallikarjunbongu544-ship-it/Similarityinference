import os
import gc
import random
import pickle
import requests
import numpy as np
import psycopg2
import tensorflow as tf
import imagehash
import cv2
import cloudinary
import cloudinary.uploader

from dotenv import load_dotenv
from urllib.parse import urlparse
from numpy.linalg import norm
from PIL import Image, ImageOps
from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from itsdangerous import URLSafeTimedSerializer
from email.mime.text import MIMEText
import smtplib

# ---------------- ENV ----------------
load_dotenv()

tf.get_logger().setLevel('ERROR')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ---------------- FLASK ----------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24))
serializer = URLSafeTimedSerializer(app.secret_key)

# ---------------- CLOUDINARY ----------------
cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("API_SECRET")
)

# ---------------- DATABASE ----------------
def get_db_connection():

    db_url = os.getenv("DATABASE_URL")

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    return psycopg2.connect(db_url)

# ---------------- MODEL ----------------
model = None

def get_model():

    global model

    if model is None:
        model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg"
        )

    return model

# ---------------- EMAIL ----------------
def send_email(to_email, subject, message):

    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")

    if not sender or not password:
        print("Email config missing")
        return

    try:

        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to_email

        server = smtplib.SMTP("smtp.gmail.com",587)
        server.starttls()
        server.login(sender,password)
        server.sendmail(sender,to_email,msg.as_string())
        server.quit()

    except Exception as e:
        print("Email error:",e)

# ---------------- IMAGE FUNCTIONS ----------------
def get_image_hash(path):

    img = Image.open(path)
    img = ImageOps.exif_transpose(img)

    return str(imagehash.phash(img))


def get_embedding(path):

    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img = img.resize((160,160))

    arr = np.array(img)
    arr = np.expand_dims(arr,axis=0)
    arr = preprocess_input(arr)

    model = get_model()

    emb = model(arr,training=False).numpy()

    return emb.flatten()


# ---------------- HIGHLIGHT ----------------
def highlight_similarity(img1,img2):

    a = cv2.imread(img1)
    b = cv2.imread(img2)

    g1 = cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(1000)

    kp1,des1 = orb.detectAndCompute(g1,None)
    kp2,des2 = orb.detectAndCompute(g2,None)

    if des1 is None or des2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1,des2,k=2)

    good=[]

    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    if len(good)<5:
        return None

    pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)

    x,y,w,h = cv2.boundingRect(pts)

    result = a.copy()

    cv2.rectangle(result,(x,y),(x+w,y+h),(0,0,255),3)

    out="highlight.jpg"
    cv2.imwrite(out,result)

    return out


# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template("login.html")


# ---------------- REGISTER ----------------
@app.route("/register",methods=["GET","POST"])
def register():

    if request.method=="POST":

        name=request.form["name"]
        email=request.form["email"]
        password=generate_password_hash(request.form["password"])

        conn=get_db_connection()
        cur=conn.cursor()

        cur.execute("SELECT * FROM users WHERE email=%s",(email,))
        if cur.fetchone():

            flash("Email already exists")
            return redirect("/register")

        cur.execute(
            "INSERT INTO users(name,email,password) VALUES(%s,%s,%s)",
            (name,email,password)
        )

        conn.commit()
        conn.close()

        flash("Registered successfully")
        return redirect("/")

    return render_template("register.html")


# ---------------- LOGIN ----------------
@app.route("/login",methods=["POST"])
def login():

    email=request.form["email"]
    password=request.form["password"]

    conn=get_db_connection()
    cur=conn.cursor()

    cur.execute(
        "SELECT name,email,password FROM users WHERE email=%s",
        (email,)
    )

    user=cur.fetchone()
    conn.close()

    if user and check_password_hash(user[2],password):

        session["user_email"]=user[1]
        session["user_name"]=user[0]
        session["role"]="user"

        return redirect("/dashboard")

    flash("Invalid login")
    return redirect("/")


# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():

    if "user_email" not in session:
        return redirect("/")

    conn=get_db_connection()
    cur=conn.cursor()

    cur.execute(
        "SELECT image_url,score,label FROM uploads WHERE user_email=%s",
        (session["user_email"],)
    )

    images=cur.fetchall()
    conn.close()

    return render_template(
        "dashboard.html",
        images=images,
        name=session["user_name"]
    )


# ---------------- UPLOAD ----------------
@app.route("/upload",methods=["GET","POST"])
def upload():

    if request.method=="GET":
        return render_template("upload.html")

    file=request.files["image"]
    label=request.form.get("label","")

    temp=f"temp_{random.randint(1000,9999)}.jpg"
    file.save(temp)

    new_hash=get_image_hash(temp)
    embedding=pickle.dumps(get_embedding(temp))

    conn=get_db_connection()
    cur=conn.cursor()

    cur.execute("""
    SELECT image_url,user_email,image_hash
    FROM uploads
    ORDER BY id DESC
    LIMIT 5
    """)

    rows=cur.fetchall()

    highest=0
    matched_user=None
    highlight_url=None

    for url,email,old_hash in rows:

        try:

            r=requests.get(url,timeout=10)

            old_temp=f"old_{random.randint(1000,9999)}.jpg"

            with open(old_temp,"wb") as f:
                f.write(r.content)

            h1=imagehash.hex_to_hash(new_hash)
            h2=imagehash.hex_to_hash(old_hash)

            score=1-((h1-h2)/64)

            if score>highest:

                highest=score
                matched_user=email

                hl=highlight_similarity(temp,old_temp)

                if hl:
                    up=cloudinary.uploader.upload(hl)
                    highlight_url=up["secure_url"]
                    os.remove(hl)

            os.remove(old_temp)

        except:
            pass

    similarity=int(highest*100)

    if similarity>=75 and matched_user!=session["user_email"]:

        send_email(
            matched_user,
            "Copyright Alert",
            f"Your image is {similarity}% similar to a new upload"
        )

        send_email(
            session["user_email"],
            "Upload Warning",
            f"Your image is {similarity}% similar to an existing image"
        )

        os.remove(temp)

        return jsonify({
            "status":"duplicate",
            "score":similarity,
            "highlight":highlight_url
        })


    up=cloudinary.uploader.upload(temp)
    url=up["secure_url"]

    cur.execute(
        "INSERT INTO uploads(image_url,user_email,score,embedding,image_hash,label) VALUES(%s,%s,%s,%s,%s,%s)",
        (url,session["user_email"],similarity,embedding,new_hash,label)
    )

    conn.commit()
    conn.close()

    os.remove(temp)

    gc.collect()

    return jsonify({
        "status":"unique",
        "score":similarity
    })


# ---------------- FORGOT PASSWORD ----------------
@app.route("/forgot_password",methods=["GET","POST"])
def forgot_password():

    if request.method=="POST":

        email=request.form["email"]

        token=serializer.dumps(email,salt="reset")

        reset=url_for(
            "reset_password",
            token=token,
            _external=True
        )

        send_email(
            email,
            "Password Reset",
            f"Reset password:\n\n{reset}"
        )

        flash("Reset email sent")

        return redirect("/")

    return render_template("forgot_password.html")


# ---------------- RESET PASSWORD ----------------
@app.route("/reset_password/<token>",methods=["GET","POST"])
def reset_password(token):

    try:
        email=serializer.loads(token,salt="reset",max_age=3600)
    except:
        flash("Invalid token")
        return redirect("/")

    if request.method=="POST":

        p=request.form["password"]
        cp=request.form["confirm_password"]

        if p!=cp:
            flash("Passwords mismatch")
            return redirect(request.url)

        hashed=generate_password_hash(p)

        conn=get_db_connection()
        cur=conn.cursor()

        cur.execute(
            "UPDATE users SET password=%s WHERE email=%s",
            (hashed,email)
        )

        conn.commit()
        conn.close()

        flash("Password updated")
        return redirect("/")

    return render_template("reset_password.html")


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():

    session.clear()
    return redirect("/")


# ---------------- RUN ----------------
if __name__=="__main__":

    port=int(os.environ.get("PORT",5000))

    app.run(host="0.0.0.0",port=port)