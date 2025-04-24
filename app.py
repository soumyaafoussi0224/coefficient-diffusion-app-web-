from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
import bcrypt
from datetime import datetime, timedelta
import secrets
from flask_mail import Mail, Message
from numpy import log as Ln, exp as e
import logging
import os
from dotenv import load_dotenv  # Import for .env usage
# Load environment variables from .env file
load_dotenv()

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')  # Load from .env
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')  # Load from .env
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')  # Load from .env
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))  # Load from .env
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True') == 'True'  # Load from .env
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')  # Load from .env
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')  # Load from .env
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')  # Add this line

db = SQLAlchemy(app)
mail = Mail(app)

# Constantes pour le calcul de diffusion
CONSTANTS = {
    'V_exp': 1.33e-05,
    'aBA': 194.5302,
    'aAB': -10.7575,
    'lambda_A': 1.127,
    'lambda_B': 0.973,
    'qA': 1.432,
    'qB': 1.4,
    'D_AB': 2.1e-5,
    'D_BA': 2.67e-5
}

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    reset_token = db.Column(db.String(100), nullable=True)
    reset_token_expiry = db.Column(db.DateTime, nullable=True)
    login_attempts = db.Column(db.Integer, default=0)
    last_attempt_time = db.Column(db.DateTime, nullable=True)
    is_locked = db.Column(db.Boolean, default=False)
    lock_until = db.Column(db.DateTime, nullable=True)

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))
    
    def is_account_locked(self):
        if self.lock_until and datetime.now() < self.lock_until:
            return True
        if self.is_locked and self.lock_until and datetime.now() >= self.lock_until:
            self.is_locked = False
            self.login_attempts = 0
            return False
        return False
    
    def increment_login_attempts(self):
        self.login_attempts += 1
        self.last_attempt_time = datetime.now()
        
        if self.login_attempts >= 5:
            self.is_locked = True
            self.lock_until = datetime.now() + timedelta(minutes=5)
        
        db.session.commit()
    
    def reset_login_attempts(self):
        self.login_attempts = 0
        self.is_locked = False
        self.lock_until = None
        db.session.commit()
    
    def generate_reset_token(self):
        self.reset_token = secrets.token_urlsafe(32)
        self.reset_token_expiry = datetime.now() + timedelta(hours=1)
        db.session.commit()
        return self.reset_token
    
    def verify_reset_token(self, token):
        if self.reset_token == token and self.reset_token_expiry > datetime.now():
            return True
        return False
    
    def clear_reset_token(self):
        self.reset_token = None
        self.reset_token_expiry = None
        db.session.commit()

# Initialisation DB
with app.app_context():
    db.create_all()

# Fonction pour le calcul de diffusion
def calcul_diffusion(Xa, T):
    if not (0 <= Xa <= 1):
        raise ValueError("La fraction Xa doit être entre 0 et 1")
    if T <= 0:
        raise ValueError("La température doit être positive")

    Xb = 1 - Xa
    phiA = (Xa * CONSTANTS['lambda_A']) / (Xa * CONSTANTS['lambda_A'] + Xb * CONSTANTS['lambda_B'])
    phiB = 1 - phiA
    tauxAB = e(-CONSTANTS['aAB'] / T)
    tauxBA = e(-CONSTANTS['aBA'] / T)
    tetaA = (Xa * CONSTANTS['qA']) / (Xa * CONSTANTS['qA'] + Xb * CONSTANTS['qB'])
    tetaB = 1 - tetaA
    tetaAA = tetaA / (tetaA + tetaB * tauxBA)
    tetaBB = tetaB / (tetaB + tetaA * tauxAB)
    tetaAB = (tetaA * tauxAB) / (tetaA * tauxAB + tetaB)
    tetaBA = (tetaB * tauxBA) / (tetaB * tauxBA + tetaA)

    termes = (
        Xb * Ln(CONSTANTS['D_AB']) +
        Xa * Ln(CONSTANTS['D_BA']) +
        2 * (Xa * Ln(Xa / phiA) + Xb * Ln(Xb / phiB)) +
        2 * Xb * Xa * (
            (phiA / Xa) * (1 - CONSTANTS['lambda_A'] / CONSTANTS['lambda_B']) +
            (phiB / Xb) * (1 - CONSTANTS['lambda_B'] / CONSTANTS['lambda_A'])
        ) +
        Xb * CONSTANTS['qA'] * (
            (1 - tetaBA ** 2) * Ln(tauxBA) +
            (1 - tetaBB ** 2) * tauxAB * Ln(tauxAB)
        ) +
        Xa * CONSTANTS['qB'] * (
            (1 - tetaAB ** 2) * Ln(tauxAB) +
            (1 - tetaAA ** 2) * tauxBA * Ln(tauxBA)
        )
    )
    solution = e(termes)
    erreur = (abs(solution - CONSTANTS['V_exp']) / CONSTANTS['V_exp']) * 100
    return {
        'lnDab': termes,
        'Dab': solution,
        'erreur': erreur,
        'Xa': Xa,
        'T': T
    }

# Routes principales
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

# Routes d'authentification
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validation des données
        if not username or not email or not password:
            flash("Tous les champs sont obligatoires", "danger")
            return redirect(url_for('register'))
            
        if password != confirm_password:
            flash("Les mots de passe ne correspondent pas", "danger")
            return redirect(url_for('register'))
            
        if len(password) < 8:
            flash("Le mot de passe doit comporter au moins 8 caractères", "danger")
            return redirect(url_for('register'))
            
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email déjà utilisé", "danger")
            return redirect(url_for('register'))

        try:
            new_user = User(username=username, email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash("Inscription réussie! Vous pouvez maintenant vous connecter", "success")
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f"Une erreur est survenue lors de l'inscription: {str(e)}", "danger")
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email).first()
        
        # Vérifier si le compte existe
        if not user:
            flash("Email ou mot de passe incorrect", "danger")
            return redirect(url_for('login'))
        
        # Vérifier si le compte est verrouillé
        if user.is_account_locked():
            remaining_time = int((user.lock_until - datetime.now()).total_seconds() / 60)
            flash(f"Compte temporairement verrouillé. Réessayez dans {remaining_time} minute(s)", "danger")
            return redirect(url_for('login'))
        
        # Vérifier le mot de passe
        if user.check_password(password):
            # Réinitialiser les tentatives en cas de succès
            user.reset_login_attempts()
            session['user_id'] = user.id
            flash("Connexion réussie", "success")
            return redirect(url_for('dashboard'))
        else:
            # Incrémenter les tentatives en cas d'échec
            user.increment_login_attempts()
            
            # Afficher le message approprié
            if user.is_locked:
                flash("Trop de tentatives. Compte temporairement verrouillé pour 5 minutes", "danger")
            else:
                remaining_attempts = 5 - user.login_attempts
                flash(f"Email ou mot de passe incorrect. Il reste {remaining_attempts} tentative(s)", "danger")
            
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        
        if user:
            try:
                # Générer un token de réinitialisation
                token = user.generate_reset_token()
                reset_link = url_for('reset_password', token=token, _external=True)
                
                # Créer le message email
                msg = Message(
                    'Réinitialisation de mot de passe',
                    recipients=[user.email],
                    body=f'''Pour réinitialiser votre mot de passe, cliquez sur le lien suivant :
{reset_link}

Ce lien expirera dans 1 heure.

Si vous n'avez pas demandé cette réinitialisation, ignorez cet email.
'''
                )
                
                # Envoyer l'email
                mail.send(msg)
                logger.info(f"Email de réinitialisation envoyé à {user.email}")
                flash("Si votre email existe dans notre système, vous recevrez un lien pour réinitialiser votre mot de passe", "info")
            except Exception as e:
                logger.error(f"Erreur lors de l'envoi d'email: {str(e)}")
                flash("Une erreur est survenue lors de l'envoi de l'email. Veuillez réessayer plus tard.", "danger")
        else:
            # Pour des raisons de sécurité, afficher le même message que si l'email existe
            logger.info(f"Tentative de réinitialisation pour un email inexistant: {email}")
            flash("Si votre email existe dans notre système, vous recevrez un lien pour réinitialiser votre mot de passe", "info")
        
        return redirect(url_for('login'))
    
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    # Trouver l'utilisateur avec ce token
    user = User.query.filter_by(reset_token=token).first()
    
    # Vérifier si le token est valide
    if not user or not user.verify_reset_token(token):
        flash("Le lien de réinitialisation est invalide ou a expiré", "danger")
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validation du mot de passe
        if password != confirm_password:
            flash("Les mots de passe ne correspondent pas", "danger")
            return redirect(url_for('reset_password', token=token))
            
        if len(password) < 8:
            flash("Le mot de passe doit comporter au moins 8 caractères", "danger")
            return redirect(url_for('reset_password', token=token))
        
        try:
            # Mettre à jour le mot de passe
            user.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            # Effacer le token
            user.clear_reset_token()
            db.session.commit()
            
            flash("Votre mot de passe a été réinitialisé avec succès", "success")
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Erreur lors de la réinitialisation du mot de passe: {str(e)}")
            flash(f"Une erreur est survenue: {str(e)}", "danger")
            return redirect(url_for('reset_password', token=token))
    
    return render_template('reset_password.html', token=token)

@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            return render_template('dashboard.html', user=user)
        # Si l'utilisateur n'existe plus, déconnecter
        session.pop('user_id', None)
    
    flash("Veuillez vous connecter", "warning")
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("Déconnexion réussie", "info")
    return redirect(url_for('index'))

# Routes pour le calculateur
@app.route('/calcul', methods=["GET", "POST"])
def calcul():
    if request.method == "POST":
        try:
            Xa = float(request.form["Xa"])
            T = float(request.form["T"])
            data = calcul_diffusion(Xa, T)
            return redirect(url_for('resultat', Xa=Xa, T=T, Dab=data['Dab'], erreur=data['erreur'], lnDab=data['lnDab']))
        except ValueError as ve:
            return render_template("calcul.html", error=str(ve))
        except Exception as e:
            return render_template("calcul.html", error=f"Erreur de calcul : {str(e)}")
    return render_template("calcul.html")

@app.route("/resultat")
def resultat():
    Xa = float(request.args.get('Xa'))
    T = float(request.args.get('T'))
    Dab = float(request.args.get('Dab'))
    erreur = float(request.args.get('erreur'))
    lnDab = float(request.args.get('lnDab'))
    
    return render_template("resultat.html", Xa=Xa, T=T, Dab=Dab, erreur=erreur, lnDab=lnDab)

@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)