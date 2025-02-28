from flask import Flask, request, jsonify, render_template, redirect, url_for, session, make_response
from joblib import load
import numpy as np
import pyrebase
import os
import json
import hashlib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from authlib.integrations.flask_client import OAuth
import ast
from datetime import datetime, timedelta


app = Flask(__name__)

# Mengatur secret key untuk sesi dengan os.urandom
app.secret_key = os.getenv('SECRET_KEY')

#Config Firebase
config = {
    'apiKey': os.getenv('FIREBASE_API_KEY'),
    'authDomain': os.getenv('FIREBASE_AUTH_DOMAIN'),
    'databaseURL': os.getenv('FIREBASE_DATABASE_URL'),
    'projectId': os.getenv('FIREBASE_PROJECT_ID'),
    'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET'),
    'messagingSenderId': os.getenv('FIREBASE_MESSAGING_SENDER_ID'),
    'appId': os.getenv('FIREBASE_APP_ID'),
    'measurementId': os.getenv('FIREBASE_MEASUREMENT_ID'),
}

# Memuat model prediksi harga mobil bekas
model = load('model_prediksi_toyota_bekas.sav')

# Memuat model deteksi mobil
model_deteksi_mobil = load_model('DetectCar.h5')

# Memuat model sentimen analisis
model_sentimen = load('sentiment_analysis_naive_bayes_model.pkl')
vectorize_text = load('vectorizer.pkl')

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

#ROUTE HALAMAN HOME
@app.route('/')
def home():
    return render_template('index.html')

#ROUTE HALAMAN LOGIN
@app.route('/login', methods=['POST', 'GET'])
def login():
    if "user" in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            user_info = auth.get_account_info(user['idToken'])
            if user_info['users'][0]['emailVerified']:
                session["user"] = email
                print("Login successful")
                return redirect(url_for('welcome'))
            else:
                error_message = "Email is not verified. Please check your email inbox and verify your email."
                return render_template('login.html', login_error=error_message)
        except Exception as e:
            print(f"Login failed: {e}")
            try:
                # Try to extract the error message from the response JSON
                error_message = 'Login Failed : ' + json.loads(e.args[1])['error']['message']
            except:
                error_message = "Failed to login due to an unknown error."
            return render_template('login.html', login_error=error_message)
    
    return render_template('login.html')

# Rute untuk logout
@app.route('/logout')
def logout():
    session.pop("user")
    return redirect(url_for('home'))


@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['forgotEmail']
        try:
            auth.send_password_reset_email(email)
            return render_template('login.html', reset_success=True)
        except Exception as e:
            print(f"Failed to send password reset email: {e}")
            return render_template('login.html', reset_success=False, error_message="Failed to send password reset email.")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if "user" in session:
        return redirect(url_for('home'))

    registration_success = False

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.create_user_with_email_and_password(email, password)
            # Send email verification
            auth.send_email_verification(user['idToken'])
            print("Register successful")
            registration_success = True
        except Exception as e:
            print(f"Failed to register: {e}")
            try:
                error_message = 'Register Failed : ' + json.loads(e.args[1])['error']['message']
            except:
                error_message = "Failed to register due to an unknown error."
            return render_template('register.html', register_error=error_message)

    return render_template('register.html', registration_success=registration_success)

@app.route('/welcome')
def welcome():
    if "user" not in session:
        return redirect(url_for('login'))

    user_email = session['user']
    db = firebase.database()
    user_data = db.child('users').child(user_email.replace('.', ',')).get().val()

    # Jika data pengguna tidak ada, buat data pengguna baru
    if not user_data:
        return render_template('welcome.html', user_email=user_email)

    if user_data:
        if 'full_name' in user_data:
            return redirect(url_for('home'))  # Jika profil sudah diatur, langsung ke halaman home
        else:
            return render_template('welcome.html', user_email=user_email)
    else:
        return redirect(url_for('login'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if "user" not in session:
        return redirect(url_for('login'))

    user_email = session['user']
    db = firebase.database()
    user_data = db.child('users').child(user_email.replace('.', ',')).get().val()

    # Initialize with default image if profile image is not available
    user_profile_image = 'https://t3.ftcdn.net/jpg/03/58/90/78/360_F_358907879_Vdu96gF4XVhjCZxN2kCG0THTsSQi8IhT.jpg'

    if user_data:
        user_profile_image = user_data.get('profile_image_url', user_profile_image)
        user_full_name = user_data.get('full_name', '')
        user_address = user_data.get('address', '')
        user_whatsapp = user_data.get('whatsApp', '')
        user_province = user_data.get('province', '')
        user_city = user_data.get('city', '')
        user_district = user_data.get('district', '')
        user_village = user_data.get('village', '')
        premium_expires = user_data.get('premium_expires', '')
    else:
        user_full_name = ''
        user_address = ''
        user_whatsapp = ''
        user_province = ''
        user_city = ''
        user_district = ''
        user_village = ''
        user_premium_expires = ''

    if request.method == 'POST':
        full_name = request.form['fullName']
        address = request.form['address']
        whatsApp = request.form['whatsApp']
        province = request.form['province']
        city = request.form['city']
        district = request.form['district']
        village = request.form['village']

        db.child('users').child(user_email.replace('.', ',')).update({
            'full_name': full_name,
            'address': address,
            'whatsApp': whatsApp,
            'province': province,
            'city': city,
            'district': district,
            'village': village
        })

        return redirect(url_for('profile'))

    return render_template('profile.html', 
                        user_profile_image=user_profile_image,
                        user_full_name=user_full_name,
                        user_address=user_address,
                        user_whatsapp=user_whatsapp,
                        user_province=user_province,
                        user_city=user_city,
                        user_district=user_district,
                        user_village=user_village,
                        user_premium_expires=premium_expires,
                        user_email=user_email) 

# Update your Flask app
@app.context_processor
def utility_processor():
    def get_profile_image():
        if "user" in session:
            user_email = session['user']
            db = firebase.database()
            user_data = db.child('users').child(user_email.replace('.', ',')).get().val()
            if user_data and 'profile_image_url' in user_data:
                return user_data['profile_image_url']
        return 'https://t3.ftcdn.net/jpg/03/58/90/78/360_F_358907879_Vdu96gF4XVhjCZxN2kCG0THTsSQi8IhT.jpg'

    def get_full_name():
        if "user" in session:
            user_email = session['user']
            db = firebase.database()
            user_data = db.child('users').child(user_email.replace('.', ',')).get().val()
            if user_data and 'full_name' in user_data:
                return user_data['full_name']
        return None
    
    def is_premium_user():
        if "user" in session:
            user_email = session['user']
            db = firebase.database()
            user_data = db.child('users').child(user_email.replace('.', ',')).get().val()
            if user_data and 'is_premium' in user_data:
                return user_data['is_premium']
        return False

    return dict(get_profile_image=get_profile_image, get_full_name=get_full_name, is_premium_user=is_premium_user)



@app.route('/edit_profile', methods=['POST'])
def edit_profile():
    if "user" not in session:
        return redirect(url_for('login'))

    user_email = session['user']

    if request.method == 'POST':
        # Mendapatkan informasi dari form
        full_name = request.form['fullName']
        address = request.form['address']
        whatsApp = request.form['whatsApp']
        province = request.form['province']
        city = request.form['city']
        district = request.form['district']
        village = request.form['village']

        # Mengupdate data pengguna di Firebase
        db = firebase.database()
        user_email = session['user']
        db.child('users').child(user_email.replace('.', ',')).update({
            'full_name': full_name,
            'address': address,
            'whatsApp': whatsApp,
            'province': province,
            'city': city,
            'district': district,
            'village': village
        })

        # Mengembalikan pengguna ke halaman profil setelah menyimpan perubahan
        return redirect(url_for('profile'))

    # Jika metode bukan POST, kembalikan ke halaman profil
    return redirect(url_for('profile'))

@app.route('/edit_profile_image', methods=['POST'])
def edit_profile_image():
    if "user" not in session:
        return redirect(url_for('login'))

    user_email = session['user']
    db = firebase.database()

    # Check if the POST request has the file part
    if 'profileImage' not in request.files:
        return redirect(url_for('profile'))

    profile_image = request.files['profileImage']

    # If user does not select file, browser also submits an empty part without filename
    if profile_image.filename == '':
        return redirect(url_for('profile'))

    # Upload image to Firebase Storage
    storage = firebase.storage()
    path = f'profile_images/{user_email}/{profile_image.filename}'
    storage.child(path).put(profile_image)

    # Get the uploaded image URL
    profile_image_url = storage.child(path).get_url(None)

    # Update user profile image URL in the database
    db.child('users').child(user_email.replace('.', ',')).update({
        'profile_image_url': profile_image_url
    })

    # Redirect back to the profile page
    return redirect(url_for('profile'))

@app.route('/delete_profile_image', methods=['POST'])
def delete_profile_image():
    if "user" not in session:
        return redirect(url_for('login'))

    user_email = session['user']
    db = firebase.database()

    # Hapus gambar dari Firebase Storage
    storage = firebase.storage()
    user_data = db.child('users').child(user_email.replace('.', ',')).get().val()

    if user_data and 'profile_image_url' in user_data:
        # Dapatkan path dari URL gambar profil
        profile_image_url = user_data['profile_image_url']
        # Dapatkan nama file dari URL (contoh: profile_images/{user_email}/{file_name})
        file_name = profile_image_url.split('/')[-1]
        path = f'profile_images/{user_email}/{file_name}'
        
        # Hapus gambar dari Firebase Storage
        try:
            storage.delete(path)
        except Exception as e:
            print(f"Failed to delete profile image: {e}")
        
        # Hapus referensi URL gambar profil dari database Firebase
        db.child('users').child(user_email.replace('.', ',')).child('profile_image_url').remove()

    return redirect(url_for('profile'))

@app.route('/predictPrice')
def predict_price():
    return render_template('predictPrice.html')

def convert_features_to_readable(features):
    human_readable = {
        'tahun': int(features[0]),
        'pajak': int(features[1]),
        'kmPerL': float(features[2]),
        'cc': float(features[3]),
        'transmisi': 'Matic' if features[4] == 1 else 'Manual',
        'bahan_bakar': 'Diesel' if features[5] == 1 else 'Electric Vehicle' if features[6] == 1 else 'Hybrid' if features[7] == 1 else 'Bensin',
        'model': 'Avanza Veloz' if features[8] == 1 else 'Calya' if features[9] == 1 else 'Corolla Cross' if features[10] == 1 else 'Fortuner' if features[11] == 1 else 'Innova' if features[12] == 1 else 'Raize' if features[13] == 1 else 'Rush',
        'kilometer': np.expm1(float(features[15]))
    }
    return human_readable

# Fungsi untuk mendapatkan kuota prediksi
@app.route('/get_quota', methods=['GET'])
def get_quota():
    if "user" in session:
        return jsonify({'quota': 'Tidak terbatas'})  # Jika pengguna sudah login
    else:
        prediction_count = request.cookies.get('prediction_count', 0)
        prediction_count = int(prediction_count)
        remaining_quota = 3 - prediction_count
        return jsonify({'quota': remaining_quota})

@app.route('/predict', methods=['POST'])
def predict():
    # Check if user is logged in
    if "user" not in session:
        prediction_count = request.cookies.get('prediction_count', 0)
        prediction_count = int(prediction_count)
        if prediction_count >= 3:
            return jsonify({'message': 'Silakan login untuk melanjutkan prediksi lebih lanjut.'}), 403

    data = request.get_json(force=True)
    features = data['features']
    
    input_data = np.array([features], dtype=float)
    prediction = model.predict(input_data)
    prediction_value = prediction.tolist()

    mpg = features[2]
    kmPerL = mpg_to_kmPerL(mpg)
    features[2] = kmPerL

    readable_features = convert_features_to_readable(features)

    if "user" in session:
        user_email = session.get('user').replace('.', ',')
        prediction_data = {
            'features': readable_features,
            'prediction': prediction_value,
            'user_email': user_email,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        db = firebase.database()
        db.child('history').push(prediction_data)
    else:
        # Update the prediction count cookie
        prediction_count += 1
        response = make_response(jsonify({'prediction': prediction_value}))
        response.set_cookie('prediction_count', str(prediction_count), max_age=60*60*24)  # Cookie valid for 1 day
        return response

    return jsonify({'prediction': prediction_value})

@app.route('/scanPrice', methods=['POST'])
def scanPrice():
    data = request.get_json(force=True)
    features = data['features']
    
    input_data = np.array([features], dtype=float)
    prediction = model.predict(input_data)
    prediction_value = prediction.tolist()

    return jsonify({'prediction': prediction_value})
    
# Fungsi untuk konversi mpg ke kmPerL
def mpg_to_kmPerL(mpg):
    return mpg / 2.35215

@app.route('/predictCar')
def predict_car():
    # Check if the user is logged in
    if "user" not in session:
        # Get the prediction count from cookies
        prediction_count = request.cookies.get('prediction_image_count', 0)
        prediction_count = int(prediction_count)
        # Calculate the remaining quota
        prediction_quota = 3 - prediction_count
    else:
        # Logged-in users have unlimited quota, set a placeholder like -1
        prediction_quota = -1

    return render_template('predictCar.html', prediction_quota=prediction_quota)

# Define a function to process the image for prediction
def process_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_input = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
    return img_input

# Function to load class map from external file
def load_class_map(file_path):
    with open(file_path, 'r') as f:
        class_map_str = f.read()
        return ast.literal_eval(class_map_str)

# Define file path for class map
class_map_file = 'class_map.txt'

# Load class map from file
class_map = load_class_map(class_map_file)

@app.route('/predict_car_brand', methods=['POST'])
def predict_car_brand():
    if "user" not in session:
        prediction_count = request.cookies.get('prediction_image_count', 0)
        prediction_count = int(prediction_count)
        if prediction_count >= 3:
            error_message = "Silakan login untuk melanjutkan prediksi lebih lanjut."
            return render_template('predictCar.html', predict_error=error_message)

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        try:
            img_array = process_image(file_path)
            prediction = model_deteksi_mobil.predict(img_array)
            predicted_class = np.argmax(prediction)

            predicted_label = class_map.get(predicted_class, 'Unknown').lower()

            predicted_model, predicted_year = predicted_label.rsplit(' ', 1)
            predicted_year = int(predicted_year)

            db = firebase.database()
            car_listings = db.child('carListings').get().val()

            filtered_listings = []
            if car_listings:
                for key, value in car_listings.items():
                    if value.get('model', '').lower() == predicted_model.lower() and value.get('tahun') >= predicted_year:
                        car = value
                        car['id'] = key
                        filtered_listings.append(car)

            session['predicted_brand'] = predicted_label
            session['car_list'] = filtered_listings

            os.remove(file_path)

            if "user" in session:
                return redirect(url_for('show_sentiment_data', page=1))
            else:
                prediction_count += 1
                response = make_response(redirect(url_for('show_sentiment_data', page=1)))
                response.set_cookie('prediction_image_count', str(prediction_count), max_age=60*60*24)  # Cookie valid for 1 day
                return response

        except Exception as e:
            os.remove(file_path)
            return str(e)

    return render_template('predictCar.html')



@app.route('/sentiment_data')
def show_sentiment_data():
    predicted_brand = session.get('predicted_brand')
    car_list = session.get('car_list', [])

    # Mengambil data sentimen dari Firebase berdasarkan merek yang diprediksi
    db = firebase.database()
    sentiment_data = db.child('database_sentiment').get().val()

    sentiment_list = []
    if sentiment_data:
        for mobil_name, sentiments in sentiment_data.items():
            if mobil_name.lower() == predicted_brand:
                sentiment_list = sentiments
                break

    # Membersihkan dan mengurutkan daftar sentimen berdasarkan published_at
    cleaned_sentiments = []
    for sentiment in sentiment_list:
        published_at = sentiment.get('published_at')
        if published_at:
            try:
                # Mengonversi tanggal/waktu ke format datetime
                if 'T' in published_at and 'Z' in published_at:
                    dt = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')
                else:
                    dt = datetime.strptime(published_at, '%Y-%m-%d %H:%M:%S')
                sentiment['published_at'] = dt
                cleaned_sentiments.append(sentiment)
            except ValueError:
                continue

    # Mengurutkan daftar sentimen berdasarkan published_at dalam urutan menurun
    sorted_sentiments = sorted(cleaned_sentiments, key=lambda x: x['published_at'], reverse=True)

    # Menghitung jumlah sentimen positif, negatif, dan netral
    sentiment_counts = {'positif': 0, 'negatif': 0, 'netral': 0}
    for sentiment in sorted_sentiments:
        klasifikasi = sentiment.get('klasifikasi', '').lower()
        if klasifikasi in sentiment_counts:
            sentiment_counts[klasifikasi] += 1

    # Pagination logic
    page = request.args.get('page', 1, type=int)
    per_page = 5
    total = len(sorted_sentiments)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_sentiments = sorted_sentiments[start:end]

    # Mengonversi kembali tanggal ke string untuk ditampilkan di template
    for sentiment in paginated_sentiments:
        sentiment['published_at'] = sentiment['published_at'].strftime('%Y-%m-%d %H:%M:%S')

    return render_template(
        'predictCar.html', 
        predicted_brand=predicted_brand, 
        car_list=car_list,
        sentiment_data=paginated_sentiments,
        sentiment_counts=sentiment_counts,
        page=page,
        total=total,
        per_page=per_page
    )


@app.route('/history')
def history():
    if "user" not in session:
        return redirect(url_for('login'))

    user_email = session['user']
    db = firebase.database()
    history_data = db.child('history').get()

    prediction_history = []
    if history_data.each():
        for prediction in history_data.each():
            prediction_val = prediction.val()
            if prediction_val['user_email'] == user_email.replace('.', ','):
                prediction_val['id'] = prediction.key()  # Menambahkan ID prediksi sebagai bagian dari data
                prediction_history.append(prediction_val)

    return render_template('history.html', prediction_history=prediction_history)



@app.route('/history/<prediction_id>')
def history_detail(prediction_id):
    if "user" not in session:
        return redirect(url_for('login'))

    user_email = session['user']
    db = firebase.database()
    prediction_ref = db.child('history').child(prediction_id).get()

    if prediction_ref.val() and prediction_ref.val()['user_email'] == user_email.replace('.', ','):
        prediction_data = prediction_ref.val()
        return render_template('history_detail.html', prediction=prediction_data)
    else:
        return "Prediction not found or unauthorized access."

@app.route('/history/<prediction_id>/delete', methods=['POST'])
def delete_prediction(prediction_id):
    if "user" not in session:
        return redirect(url_for('login'))

    user_email = session['user']
    db = firebase.database()
    prediction_ref = db.child('history').child(prediction_id).get()

    if prediction_ref.val() and prediction_ref.val()['user_email'] == user_email.replace('.', ','):
        # Hapus prediksi dari database
        db.child('history').child(prediction_id).remove()
        return redirect(url_for('history'))
    else:
        return "Prediction not found or unauthorized access."
    
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    if "user" not in session:
        return redirect(url_for('login'))

    sentiment_text = request.form['sentiment']
    
    # Preprocess sentiment text
    sentiment_vector = vectorize_text.transform([sentiment_text])  # Use transform instead of vectorize_text

    # Predict sentiment
    predicted_sentimen = model_sentimen.predict(sentiment_vector)[0]
    
    if predicted_sentimen == 2:
        sentiment_label = "Positif"
    elif predicted_sentimen == 1:
        sentiment_label = "Netral"
    elif predicted_sentimen == 0:
        sentiment_label = "Negatif"

    # Save sentiment data to Firebase
    user_email = session['user'].replace('.', ',')
    db = firebase.database()

    user_data = db.child('users').child(user_email).get().val()

    # Get full name from user data if exists
    author_name = "@" + user_data.get('full_name', user_email)

    predicted_brand = session.get('predicted_brand', 'unknown').capitalize()
       # Get the last ID
    sentiments = db.child('database_sentiment').child(predicted_brand).order_by_key().get()
    last_id = 0
    if sentiments.each():
        last_id = max(int(sentiment.key()) for sentiment in sentiments.each())

    new_id = last_id + 1

    sentiment_data = {
        'author': author_name,
        'author_email' : user_email,
        'published_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'text': sentiment_text,
        'klasifikasi': sentiment_label
    }
    db.child('database_sentiment').child(predicted_brand).child(str(new_id)).set(sentiment_data)

    return redirect(url_for('show_sentiment_data', page=1))

@app.route('/carListing')
def car_listing():
    return render_template('carListing.html')

@app.route('/upgrade', methods=['GET'])
def upgrade():
    plan = request.args.get('plan')
    if plan == '1month' or plan == '1year':
        if 'user' in session:
            user_email = session['user']
            db = firebase.database()
            user_email = session['user']
            if plan == '1month':
                is_premium = True
                premium_expires = (datetime.now() + timedelta(days=30)).isoformat()
            elif plan == '1year':
                is_premium = True
                premium_expires = (datetime.now() + timedelta(days=365)).isoformat()
            else:
                return jsonify({'error': 'Invalid plan'}), 400
            db.child('users').child(user_email.replace('.', ',')).update({
                'is_premium': is_premium,
                'premium_expires': premium_expires
            })
            message = f'Upgrade to {plan.capitalize()} Premium successful'
            return render_template('carListing.html', message=message)
        else:
            message ='User not authenticated'
            return render_template('carListing.html', message=message)
    else:
        message ='Invalid plan'
        return render_template('carListing.html', message=message)

@app.route('/predictCarPrice', methods=['POST'])
def predict_car_price():
    data = request.get_json(force=True)
    features = data['features']
    # Konversi fitur ke numpy array dengan bentuk yang benar
    input_data = np.array([features], dtype=float)
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/submitCarListing', methods=['POST'])
def submit_car_listing():
    if request.method == 'POST':
        # Mendapatkan informasi dari form
        tahun = request.form['tahun']
        kilometer = request.form['kilometer']
        pajak = request.form['pajak']
        kmPerL = request.form['kmPerL']
        cc = request.form['cc']
        transmisi = request.form['transmisi']
        bahan_bakar = request.form['bahan_bakar']
        model = request.form['model']
        carPrice = request.form['carPrice']
        carDescription = request.form['carDescription']
        user_email = session.get('user').replace('.', ',')

        # Handle multiple image uploads
        images = []
        for i in range(1, 4):  # Assuming you have carImages, carImages2, carImages3
            image = request.files.get(f'carImages{i}')
            if image:
                image_filename = image.filename
                hash_object = hashlib.sha256(image_filename.encode())
                hex_dig = hash_object.hexdigest()
                encrypted_filename = f"{hex_dig}.jpg"

                # Save image to Firebase Storage
                storage = firebase.storage()
                path = f"car_images/{encrypted_filename}"
                storage.child(path).put(image)

                # Get the uploaded image URL
                car_image_url = storage.child(path).get_url(None)
                images.append(car_image_url)

        
        # Create a new car listing object
        new_listing = {
            'tahun': int(tahun),
            'kilometer': kilometer,
            'pajak': pajak,
            'kmPerL': kmPerL,
            'cc': cc,
            'transmisi': transmisi,
            'bahan_bakar': bahan_bakar,
            'model': model,
            'carPrice': carPrice,
            'carDescription': carDescription,
            'user_email': user_email,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Add image URLs to the listing if any images were uploaded
        if images:
            new_listing['imageURLs'] = images

        # Push the new listing to Firebase
        db = firebase.database()
        db.child('carListings').push(new_listing)

        return redirect(url_for('car_listing'))

    # Handle other HTTP methods or invalid requests
    return redirect(url_for('error_page'))

@app.route('/edit/<car_id>', methods=['POST'])
def edit_car(car_id):
    # Get form data
    model = request.form['model']
    bahan_bakar = request.form['bahan_bakar']
    tahun = request.form['tahun']
    kilometer = request.form['kilometer']
    pajak = request.form['pajak']
    kmPerL = request.form['kmPerL']
    cc = request.form['cc']
    transmisi = request.form['transmisi']
    carPrice = request.form['carPrice']
    carDescription = request.form['carDescription']
    
    # Handle multiple image uploads
    images = []
    for i in range(1, 4):  # Assuming you have carImages, carImages2, carImages3
        image = request.files.get(f'carImages{i}')
        if image:
            image_filename = image.filename
            hash_object = hashlib.sha256(image_filename.encode())
            hex_dig = hash_object.hexdigest()
            encrypted_filename = f"{hex_dig}.jpg"

            # Save image to Firebase Storage
            storage = firebase.storage()
            path = f"car_images/{encrypted_filename}"
            storage.child(path).put(image)

            # Get the uploaded image URL
            car_image_url = storage.child(path).get_url(None)
            images.append(car_image_url)

    # Create a new car listing object
    edit_listing = {
        'tahun': int(tahun),
        'kilometer': kilometer,
        'pajak': pajak,
        'kmPerL': kmPerL,
        'cc': cc,
        'transmisi': transmisi,
        'bahan_bakar': bahan_bakar,
        'model': model,
        'carPrice': carPrice,
        'carDescription': carDescription,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Add image URLs to the listing if any images were uploaded
    if images:
        edit_listing['imageURLs'] = images

    

    # Update the car listing in Firebase
    db = firebase.database()
    db.child('carListings').child(car_id).update(edit_listing)

    return redirect(url_for('car_listing'))


@app.route('/carListings', methods=['GET'])
def get_car_listings():
    db = firebase.database()
    car_listings = db.child('carListings').get().val()
    
    # Mengambil query pencarian dari permintaan GET
    search_query = request.args.get('search', '').lower()

    # Konversi respons Firebase menjadi daftar
    if car_listings:
        car_listings = [{**value, 'id': key} for key, value in car_listings.items()]
    else:
        car_listings = []

    user_listings = []
    general_listings = []

    if 'user' in session:
        user_email = session['user'].replace('.', ',')
        user_listings = [car for car in car_listings if car.get('user_email') == user_email]
        general_listings = [car for car in car_listings if car.get('user_email') != user_email]
    else:
        general_listings = car_listings

    # Filter berdasarkan pencarian
    if search_query:
        user_listings = [car for car in user_listings if search_query in car['model'].lower()]
        general_listings = [car for car in general_listings if search_query in car['model'].lower()]

    return jsonify({
        'user_listings': user_listings,
        'general_listings': general_listings
    })

@app.route('/carListings/<car_id>')
def car_details(car_id):

    db = firebase.database()
    car_data = db.child('carListings').child(car_id).get().val()

    if car_data:
        if 'views' in car_data:
            car_data['views'] += 1
        else:
            car_data['views'] = 1
        db.child('carListings').child(car_id).update({'views': car_data['views']})
    
        car_data['id'] = car_id  # Add this line to include car ID in car_data
        user_email = car_data['user_email']
        user_data = db.child('users').child(user_email.replace('.', ',')).get().val()
        
        if user_data:
            return render_template('carDetails.html', car=car_data, user=user_data)
        else:
            return "User not found", 404
    else:
        return "Car not found", 404
    

@app.route('/delete/<car_id>')
def delete_car(car_id):
    # Delete the car listing from Firebase
    db = firebase.database()
    user_email = session.get('user').replace('.', ',')
    car = db.child('carListings').child(car_id).get().val()
    
    if car['user_email'] == user_email:
        db.child('carListings').child(car_id).remove()
        return redirect(url_for('car_listing', message='Car listing deleted successfully.', message_type='success'))
    else:
        return redirect(url_for('car_listing', message='You are not authorized to delete this car listing.', message_type='danger'))




if __name__ == '__main__':
    app.run(debug=True)
