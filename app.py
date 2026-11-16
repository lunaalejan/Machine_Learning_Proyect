import os
import math
from datetime import date, timedelta
from functools import wraps

import click

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    abort
)

from flask_login import (
    LoginManager,
    login_required,
    login_user,
    logout_user,
    current_user
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from sqlalchemy.exc import IntegrityError

from database import db
from database.models import (
    User,
    Equipment,
    MaintenanceRecord,
    Alert,
    Prediction
)

from machine_learning.predict import predict_failure


app = Flask(__name__)

# ==============================
# APPLICATION CONFIGURATION
# ==============================

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "dev-only-change-this-secret-before-deployment"
)

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    "sqlite:///predictive_maintenance.db"
).replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)


# ==============================
# FLASK-LOGIN CONFIGURATION
# ==============================

login_manager = LoginManager()
login_manager.init_app(app)

login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "warning"


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ==============================
# ROLE-BASED ACCESS CONTROL
# ==============================

def role_required(*allowed_roles):
    """
    Restrict access to users with specific roles.
    """

    def decorator(view_function):

        @wraps(view_function)
        @login_required
        def wrapped_view(*args, **kwargs):

            if current_user.role not in allowed_roles:
                abort(403)

            return view_function(*args, **kwargs)

        return wrapped_view

    return decorator


def require_role_for_post(*allowed_roles):
    """
    Allow authenticated users to view a page,
    but restrict POST operations to selected roles.
    """

    if (
        request.method == "POST"
        and current_user.role not in allowed_roles
    ):
        abort(403)


# ==============================
# AUTHENTICATION
# ==============================

@app.route("/login", methods=["GET", "POST"])
def login():

    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    error = None

    if request.method == "POST":

        email = request.form.get(
            "email", ""
        ).strip().lower()

        password = request.form.get(
            "password", ""
        )

        user = User.query.filter(
            db.func.lower(User.email) == email
        ).first()

        if user and check_password_hash(
            user.password_hash,
            password
        ):

            login_user(user)

            next_page = request.args.get("next")

            if (
                next_page
                and next_page.startswith("/")
                and not next_page.startswith("//")
            ):
                return redirect(next_page)

            return redirect(url_for("dashboard"))

        error = "Invalid email or password."

    return render_template(
        "login.html",
        error=error,
        active_page="login"
    )


@app.route("/register", methods=["GET", "POST"])
def register():

    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    error = None

    if request.method == "POST":

        full_name = request.form.get(
            "full_name", ""
        ).strip()

        email = request.form.get(
            "email", ""
        ).strip().lower()

        password = request.form.get(
            "password", ""
        )

        confirm_password = request.form.get(
            "confirm_password", ""
        )

        try:

            if not full_name or not email:
                raise ValueError(
                    "Full name and email are required."
                )

            if len(full_name) > 120:
                raise ValueError(
                    "Full name is too long."
                )

            if len(email) > 150 or "@" not in email:
                raise ValueError(
                    "Please enter a valid email address."
                )

            if len(password) < 8:
                raise ValueError(
                    "Password must contain at least 8 characters."
                )

            if password != confirm_password:
                raise ValueError(
                    "Passwords do not match."
                )

            existing_user = User.query.filter(
                db.func.lower(User.email) == email
            ).first()

            if existing_user:
                raise ValueError(
                    "An account with this email already exists."
                )

            new_user = User(
                full_name=full_name,
                email=email,
                password_hash=generate_password_hash(password),
                role="technician"
            )

            db.session.add(new_user)
            db.session.commit()

            login_user(new_user)

            return redirect(url_for("dashboard"))

        except ValueError as exc:
            db.session.rollback()
            error = str(exc)

        except IntegrityError:
            db.session.rollback()
            error = "This email is already registered."

        except Exception:
            db.session.rollback()
            app.logger.exception(
                "User registration failed"
            )
            error = (
                "An unexpected error occurred during registration."
            )

    return render_template(
        "register.html",
        error=error,
        active_page="register"
    )


@app.route("/logout", methods=["GET", "POST"])
@login_required
def logout():

    logout_user()

    return redirect(url_for("login"))


# ==============================
# CREATE FIRST ADMINISTRATOR
# ==============================

@app.cli.command("create-admin")
def create_admin():
    """Create or update an administrator account."""

    email = click.prompt(
        "Administrator email"
    ).strip().lower()

    full_name = click.prompt(
        "Administrator full name"
    ).strip()

    password = click.prompt(
        "Administrator password",
        hide_input=True,
        confirmation_prompt=True
    )

    if not email or "@" not in email:
        click.echo("Please enter a valid email.")
        return

    if len(password) < 8:
        click.echo(
            "Password must contain at least 8 characters."
        )
        return

    existing_user = User.query.filter(
        db.func.lower(User.email) == email
    ).first()

    if existing_user:

        existing_user.role = "admin"
        existing_user.password_hash = generate_password_hash(
            password
        )
        existing_user.full_name = full_name

        db.session.commit()

        click.echo(
            "Existing account updated to administrator."
        )

    else:

        admin_user = User(
            full_name=full_name,
            email=email,
            password_hash=generate_password_hash(password),
            role="admin"
        )

        db.session.add(admin_user)
        db.session.commit()

        click.echo(
            "Administrator account created successfully."
        )


# ==============================
# USER MANAGEMENT - ADMIN ONLY
# ==============================

@app.route("/users", methods=["GET", "POST"])
@role_required("admin")
def user_management():

    error = None
    success = None

    if request.method == "POST":

        try:

            full_name = request.form.get(
                "full_name", ""
            ).strip()

            email = request.form.get(
                "email", ""
            ).strip().lower()

            password = request.form.get(
                "password", ""
            )

            role = request.form.get(
                "role", ""
            ).strip().lower()

            allowed_roles = [
                "admin",
                "supervisor",
                "technician"
            ]

            if not full_name or not email or not password:
                raise ValueError(
                    "All fields are required."
                )

            if len(full_name) > 120:
                raise ValueError(
                    "Full name is too long."
                )

            if len(email) > 150 or "@" not in email:
                raise ValueError(
                    "Please enter a valid email address."
                )

            if len(password) < 8:
                raise ValueError(
                    "Password must contain at least 8 characters."
                )

            if role not in allowed_roles:
                raise ValueError(
                    "Please select a valid user role."
                )

            existing_user = User.query.filter(
                db.func.lower(User.email) == email
            ).first()

            if existing_user:
                raise ValueError(
                    "This email is already registered."
                )

            new_user = User(
                full_name=full_name,
                email=email,
                password_hash=generate_password_hash(password),
                role=role
            )

            db.session.add(new_user)
            db.session.commit()

            return redirect(
                url_for("user_management", created=1)
            )

        except ValueError as exc:
            db.session.rollback()
            error = str(exc)

        except IntegrityError:
            db.session.rollback()
            error = "This email is already registered."

        except Exception:
            db.session.rollback()
            app.logger.exception(
                "Administrator user creation failed"
            )
            error = "An unexpected error occurred."

    users = User.query.order_by(
        User.created_at.desc()
    ).all()

    if request.args.get("created") == "1":
        success = "User created successfully!"

    if request.args.get("updated") == "1":
        success = "User role updated successfully!"

    return render_template(
        "users.html",
        users=users,
        error=error,
        success=success,
        active_page="users"
    )


@app.route(
    "/users/<int:user_id>/role",
    methods=["POST"]
)
@role_required("admin")
def update_user_role(user_id):

    user = db.session.get(User, user_id)

    if user is None:
        abort(404)

    new_role = request.form.get(
        "role", ""
    ).strip().lower()

    allowed_roles = [
        "admin",
        "supervisor",
        "technician"
    ]

    if new_role not in allowed_roles:
        abort(400)

    if user.id == current_user.id:
        abort(403)

    if user.role == "admin" and new_role != "admin":

        admin_count = User.query.filter_by(
            role="admin"
        ).count()

        if admin_count <= 1:
            abort(403)

    user.role = new_role
    db.session.commit()

    return redirect(
        url_for("user_management", updated=1)
    )


# ==============================
# HOME PAGE - PUBLIC
# ==============================

@app.route("/")
def home():

    return render_template(
        "index.html",
        active_page="home"
    )


# ==============================
# PGC PAGE - PUBLIC
# ==============================

@app.route("/pgc")
def pgc():

    return render_template(
        "pgc.html",
        active_page="pgc"
    )


# ==============================
# CRISP-DM METHODOLOGY - PUBLIC
# ==============================

@app.route("/crisp-dm")
def crisp_dm():

    return render_template(
        "crisp_dm.html",
        active_page="crisp_dm"
    )


# ==============================
# PHASE 1 - BUSINESS AND DATA UNDERSTANDING
# ==============================

@app.route("/business-data-understanding")
def business_data_understanding():

    return render_template(
        "business_data_understanding.html",
        active_page="crisp_dm"
    )


# ==============================
# PHASE 2 - DATA PREPARATION
# ==============================

@app.route("/data-preparation")
def data_preparation():

    return render_template(
        "data_preparation.html",
        active_page="crisp_dm"
    )


# ==============================
# PHASE 3 - MODELING
# ==============================

@app.route("/modeling")
def modeling():

    return render_template(
        "modeling.html",
        active_page="crisp_dm"
    )


# ==============================
# PHASE 4 - EVALUATION
# ==============================

@app.route("/evaluation")
def evaluation():

    return render_template(
        "evaluation.html",
        active_page="crisp_dm"
    )


# ==============================
# PHASE 5 - DEPLOYMENT
# ==============================

@app.route("/deployment")
def deployment():

    return render_template(
        "deployment.html",
        active_page="crisp_dm"
    )


# ==============================
# PHASE 6 - MONITORING AND MAINTENANCE
# ==============================

@app.route("/monitoring_maintenance")
def monitoring_maintenance():

    return render_template(
        "monitoring_maintenance.html",
        active_page="crisp_dm"
    )


# ==============================
# EQUIPMENT REGISTRATION
# ==============================

@app.route("/application", methods=["GET", "POST"])
@login_required
def application():

    require_role_for_post("admin", "technician")

    success = None
    error = None

    if request.method == "POST":

        try:

            name = request.form.get(
                "name", ""
            ).strip()

            equipment_type = request.form.get(
                "equipment_type", ""
            ).strip()

            brand = request.form.get(
                "brand", ""
            ).strip()

            model = request.form.get(
                "model", ""
            ).strip()

            serial_number = request.form.get(
                "serial_number", ""
            ).strip()

            location = request.form.get(
                "location", ""
            ).strip()

            interval_value = request.form.get(
                "maintenance_interval_days", "90"
            ).strip()

            installation_value = request.form.get(
                "installation_date", ""
            ).strip()

            if not name or not equipment_type:
                raise ValueError(
                    "Equipment name and type are required."
                )

            interval = int(interval_value)

            if interval < 1:
                raise ValueError(
                    "Maintenance interval must be at least 1 day."
                )

            installation_date = (
                date.fromisoformat(installation_value)
                if installation_value else None
            )

            if (
                installation_date
                and installation_date > date.today()
            ):
                raise ValueError(
                    "Installation date cannot be in the future."
                )

            if serial_number:

                existing_equipment = Equipment.query.filter_by(
                    serial_number=serial_number
                ).first()

                if existing_equipment:
                    raise ValueError(
                        "This serial number is already registered."
                    )

            new_equipment = Equipment(
                name=name,
                equipment_type=equipment_type,
                brand=brand or None,
                model=model or None,
                serial_number=serial_number or None,
                location=location or None,
                installation_date=installation_date,
                maintenance_interval_days=interval,
                status="active",
                owner_id=current_user.id
            )

            db.session.add(new_equipment)
            db.session.commit()

            return redirect(
                url_for("application", registered=1)
            )

        except ValueError as exc:
            db.session.rollback()
            error = str(exc)

        except IntegrityError:
            db.session.rollback()
            error = (
                "This equipment could not be registered. "
                "Check whether the serial number already exists."
            )

        except Exception:
            db.session.rollback()
            app.logger.exception(
                "Equipment registration failed"
            )
            error = (
                "An unexpected error occurred. "
                "Please check the information and try again."
            )

    if request.args.get("registered") == "1":
        success = "Equipment registered successfully!"

    equipment = Equipment.query.order_by(
        Equipment.id.desc()
    ).all()

    return render_template(
        "application.html",
        equipment=equipment,
        success=success,
        error=error,
        active_page="application"
    )


# ==============================
# AUTOMATIC ALERT MANAGEMENT
# ==============================

def update_automatic_alerts(
    equipment,
    desired_alert_type,
    message
):
    """
    Synchronize automatic maintenance alerts.
    Keep only one active alert per equipment and alert type.
    """

    automatic_types = [
        "maintenance_due_soon",
        "maintenance_overdue"
    ]

    active_alerts = Alert.query.filter(
        Alert.equipment_id == equipment.id,
        Alert.status == "active",
        Alert.alert_type.in_(automatic_types)
    ).all()

    matching_alerts = [
        alert for alert in active_alerts
        if alert.alert_type == desired_alert_type
    ]

    changed = False

    for alert in active_alerts:

        if (
            desired_alert_type is None
            or alert.alert_type != desired_alert_type
        ):
            alert.status = "resolved"
            changed = True

    if desired_alert_type:

        if matching_alerts:

            current_alert = matching_alerts[0]

            if current_alert.message != message:
                current_alert.message = message
                changed = True

            for duplicate in matching_alerts[1:]:
                duplicate.status = "resolved"
                changed = True

        else:

            new_alert = Alert(
                equipment_id=equipment.id,
                alert_type=desired_alert_type,
                message=message,
                status="active",
                is_read=False
            )

            db.session.add(new_alert)
            changed = True

    return changed


# ==============================
# CALCULATE MAINTENANCE STATUS
# ==============================

def calculate_maintenance_status(equipment, today):
    """
    Calculate next maintenance date, remaining days
    and maintenance status for one equipment.
    """

    reference_date = (
        equipment.last_maintenance_date
        or equipment.installation_date
    )

    if reference_date is None:

        return {
            "reference_date": None,
            "next_maintenance": None,
            "days_remaining": None,
            "status": "No Date",
            "alert_type": None,
            "message": ""
        }

    next_maintenance = reference_date + timedelta(
        days=equipment.maintenance_interval_days
    )

    days_remaining = (
        next_maintenance - today
    ).days

    if days_remaining < 0:

        status = "Overdue"
        alert_type = "maintenance_overdue"

        message = (
            f"Maintenance overdue for {equipment.name}. "
            f"It was due on {next_maintenance}."
        )

    elif days_remaining <= 7:

        status = "Due Soon"
        alert_type = "maintenance_due_soon"

        message = (
            f"Maintenance due soon for {equipment.name}. "
            f"Days remaining: {days_remaining}."
        )

    else:

        status = "Up to Date"
        alert_type = None
        message = ""

    return {
        "reference_date": reference_date,
        "next_maintenance": next_maintenance,
        "days_remaining": days_remaining,
        "status": status,
        "alert_type": alert_type,
        "message": message
    }


# ==============================
# SYNCHRONIZE ALL AUTOMATIC ALERTS
# ==============================

def synchronize_all_automatic_alerts():

    today = date.today()
    alerts_changed = False

    equipment_list = Equipment.query.all()

    for item in equipment_list:

        result = calculate_maintenance_status(
            item,
            today
        )

        changed = update_automatic_alerts(
            item,
            result["alert_type"],
            result["message"]
        )

        if changed:
            alerts_changed = True

    if alerts_changed:
        db.session.commit()


# ==============================
# MAINTENANCE DASHBOARD
# ==============================

@app.route("/dashboard")
@login_required
def dashboard():

    synchronize_all_automatic_alerts()

    total_equipment = Equipment.query.count()
    total_maintenance = MaintenanceRecord.query.count()
    total_predictions = Prediction.query.count()

    total_active_alerts = Alert.query.filter_by(
        status="active"
    ).count()

    low_risk = Prediction.query.filter_by(
        risk_level="Low"
    ).count()

    medium_risk = Prediction.query.filter_by(
        risk_level="Medium"
    ).count()

    high_risk = Prediction.query.filter_by(
        risk_level="High"
    ).count()

    latest_predictions = (
        Prediction.query
        .order_by(Prediction.predicted_at.desc())
        .limit(5)
        .all()
    )

    return render_template(
        "dashboard.html",
        active_page="dashboard",
        total_equipment=total_equipment,
        total_maintenance=total_maintenance,
        total_predictions=total_predictions,
        total_active_alerts=total_active_alerts,
        low_risk=low_risk,
        medium_risk=medium_risk,
        high_risk=high_risk,
        latest_predictions=latest_predictions
    )


# ==============================
# MAINTENANCE MANAGEMENT
# ==============================

@app.route("/maintenance", methods=["GET", "POST"])
@login_required
def maintenance():

    require_role_for_post("admin", "technician")

    success = None
    error = None

    if request.method == "POST":

        try:

            equipment_id = request.form.get(
                "equipment_id",
                type=int
            )

            maintenance_date_value = request.form.get(
                "maintenance_date",
                ""
            ).strip()

            if not maintenance_date_value:
                raise ValueError(
                    "Maintenance date is required."
                )

            maintenance_date = date.fromisoformat(
                maintenance_date_value
            )

            if maintenance_date > date.today():
                raise ValueError(
                    "Maintenance date cannot be in the future."
                )

            maintenance_type = request.form.get(
                "maintenance_type",
                ""
            ).strip()

            description = request.form.get(
                "description",
                ""
            ).strip()

            performed_by = request.form.get(
                "performed_by",
                ""
            ).strip()

            cost_value = request.form.get(
                "cost",
                ""
            ).strip()

            selected_equipment = db.session.get(
                Equipment,
                equipment_id
            )

            if selected_equipment is None:
                raise ValueError(
                    "Please select a valid equipment."
                )

            if maintenance_type not in [
                "Preventive",
                "Corrective"
            ]:
                raise ValueError(
                    "Please select a valid maintenance type."
                )

            cost = float(cost_value) if cost_value else None

            if cost is not None:

                if not math.isfinite(cost) or cost < 0:
                    raise ValueError(
                        "Maintenance cost must be a valid "
                        "non-negative number."
                    )

            new_record = MaintenanceRecord(
                equipment_id=equipment_id,
                maintenance_date=maintenance_date,
                maintenance_type=maintenance_type,
                description=description or None,
                performed_by=performed_by or None,
                cost=cost
            )

            db.session.add(new_record)

            if (
                selected_equipment.last_maintenance_date is None
                or maintenance_date >= selected_equipment.last_maintenance_date
            ):
                selected_equipment.last_maintenance_date = (
                    maintenance_date
                )

            db.session.commit()

            return redirect(
                url_for("maintenance", saved=1)
            )

        except ValueError as exc:
            db.session.rollback()
            error = str(exc)

        except Exception:
            db.session.rollback()
            app.logger.exception(
                "Maintenance registration failed"
            )
            error = (
                "An error occurred while saving "
                "the maintenance record."
            )

    equipment_list = Equipment.query.order_by(
        Equipment.name.asc()
    ).all()

    records = MaintenanceRecord.query.order_by(
        MaintenanceRecord.maintenance_date.desc()
    ).all()

    maintenance_schedule = []

    today = date.today()

    for item in equipment_list:

        result = calculate_maintenance_status(
            item,
            today
        )

        maintenance_schedule.append({
            "equipment": item,
            "reference_date": result["reference_date"],
            "next_maintenance": result["next_maintenance"],
            "days_remaining": result["days_remaining"],
            "status": result["status"]
        })

        update_automatic_alerts(
            item,
            result["alert_type"],
            result["message"]
        )

    db.session.commit()

    if request.args.get("saved") == "1":
        success = (
            "Maintenance record registered successfully!"
        )

    return render_template(
        "maintenance.html",
        equipment=equipment_list,
        records=records,
        maintenance_schedule=maintenance_schedule,
        success=success,
        error=error,
        active_page="maintenance"
    )


# ==============================
# MACHINE LEARNING PREDICTIONS
# ==============================

@app.route("/predictions", methods=["GET", "POST"])
@login_required
def predictions():

    result = None
    error = None
    success = None
    selected_equipment_id = None

    if request.method == "POST":

        try:

            equipment_id = request.form.get(
                "equipment_id",
                type=int
            )

            selected_equipment_id = equipment_id

            selected_equipment = db.session.get(
                Equipment,
                equipment_id
            )

            if selected_equipment is None:
                raise ValueError(
                    "Please select valid registered equipment."
                )

            machine_type = request.form.get(
                "machine_type",
                ""
            ).strip()

            if machine_type not in ["L", "M", "H"]:
                raise ValueError(
                    "Please select a valid machine type."
                )

            air_temperature = float(
                request.form.get("air_temperature", "")
            )

            process_temperature = float(
                request.form.get("process_temperature", "")
            )

            rotational_speed = float(
                request.form.get("rotational_speed", "")
            )

            torque = float(
                request.form.get("torque", "")
            )

            tool_wear = float(
                request.form.get("tool_wear", "")
            )

            values = [
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear
            ]

            if not all(
                math.isfinite(value)
                for value in values
            ):
                raise ValueError(
                    "Please enter valid numeric values."
                )

            if rotational_speed <= 0:
                raise ValueError(
                    "Rotational speed must be greater than zero."
                )

            if torque < 0 or tool_wear < 0:
                raise ValueError(
                    "Torque and tool wear cannot be negative."
                )

            equipment_data = {
                "Type": machine_type,
                "Air temperature [K]": air_temperature,
                "Process temperature [K]": process_temperature,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear
            }

            result = predict_failure(equipment_data)

            new_prediction = Prediction(
                equipment_id=selected_equipment.id,
                failure_probability=result["failure_probability"],
                risk_level=result["risk_level"],
                model_version=result["model_version"]
            )

            db.session.add(new_prediction)
            db.session.commit()

        except (ValueError, TypeError) as exc:
            db.session.rollback()
            error = str(exc)

        except FileNotFoundError as exc:
            db.session.rollback()
            error = str(exc)

        except Exception:
            db.session.rollback()
            app.logger.exception(
                "Failure prediction failed"
            )
            error = (
                "An unexpected error occurred while "
                "generating the prediction."
            )

    # Show the success message after an administrator deletes a prediction.
    if request.args.get("deleted") == "1":
        success = "Prediction deleted successfully!"

    equipment_list = Equipment.query.order_by(
        Equipment.name.asc()
    ).all()

    prediction_history = (
        Prediction.query
        .order_by(Prediction.predicted_at.desc())
        .all()
    )

    return render_template(
        "predictions.html",
        equipment=equipment_list,
        predictions=prediction_history,
        result=result,
        error=error,
        success=success,
        selected_equipment_id=selected_equipment_id,
        active_page="predictions"
    )


# ==============================
# DELETE PREDICTION - ADMIN ONLY
# ==============================

@app.route(
    "/predictions/<int:prediction_id>/delete",
    methods=["POST"]
)
@role_required("admin")
def delete_prediction(prediction_id):
    """
    Only administrators can delete prediction records.
    Supervisors and technicians receive HTTP 403.
    """

    prediction = db.session.get(
        Prediction,
        prediction_id
    )

    if prediction is None:
        abort(404)

    try:

        db.session.delete(prediction)
        db.session.commit()

    except Exception:
        db.session.rollback()
        app.logger.exception(
            "Prediction deletion failed"
        )
        abort(500)

    return redirect(
        url_for("predictions", deleted=1)
    )


# ==============================
# ALERT MANAGEMENT PAGE
# ==============================

@app.route("/alerts")
@login_required
def alerts():

    synchronize_all_automatic_alerts()

    all_alerts = Alert.query.order_by(
        Alert.created_at.desc()
    ).all()

    success = None

    if request.args.get("read") == "1":
        success = "Alert marked as read successfully!"

    return render_template(
        "alerts.html",
        alerts=all_alerts,
        success=success,
        active_page="alerts"
    )


# ==============================
# MARK ALERT AS READ
# ==============================

@app.route(
    "/alerts/<int:alert_id>/read",
    methods=["POST"]
)
@role_required("admin", "supervisor", "technician")
def mark_alert_read(alert_id):

    alert = db.session.get(Alert, alert_id)

    if alert is None:
        return redirect(url_for("alerts"))

    alert.is_read = True
    db.session.commit()

    return redirect(
        url_for("alerts", read=1)
    )


# ==============================
# FORBIDDEN ACCESS HANDLER
# ==============================

@app.errorhandler(403)
def forbidden(error):

    return (
        render_template(
            "403.html",
            active_page="forbidden"
        ),
        403
    )


# ==============================
# CREATE DATABASE TABLES
# ==============================

with app.app_context():
    db.create_all()


# ==============================
# RUN APPLICATION
# ==============================

if __name__ == "__main__":

    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1"
    )