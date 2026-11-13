from datetime import date, timedelta
import math

from flask import (
    Flask,
    render_template,
    render_template_string,
    request,
    redirect,
    url_for
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
# DATABASE CONFIGURATION
# ==============================

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///predictive_maintenance.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def home():

    return render_template(
        "index.html",
        active_page="home"
    )


# ==============================
# PGC PAGE
# ==============================

@app.route("/pgc")
def pgc():
    return render_template(
        "pgc.html",
        active_page="pgc"
    )


# ==============================
# CRISP-DM PAGE
# ==============================


# ==============================
# CRISP-DM METHODOLOGY
# ==============================

@app.route("/crisp-dm")
def crisp_dm():

    return render_template(
        "crisp_dm.html",
        active_page="crisp_dm"
    )


# ==============================
# PHASE 1 - BUSINESS UNDERSTANDING
# ==============================

@app.route("/business-understanding")
def business_understanding():

    return render_template(
        "business_understanding.html",
        active_page="crisp_dm"
    )


# ==============================
# PHASE 2 - DATA UNDERSTANDING
# ==============================

@app.route("/data-understanding")
def data_understanding():

    return render_template(
        "data_understanding.html",
        active_page="crisp_dm"
    )


# ==============================
# PHASE 3 - DATA PREPARATION
# ==============================

@app.route("/data-preparation")
def data_preparation():

    return render_template(
        "data_preparation.html",
        active_page="crisp_dm"
    )


# ==============================
# PHASE 4 - MODELING
# ==============================

@app.route("/modeling")
def modeling():

    return render_template(
        "modeling.html",
        active_page="crisp_dm"
    )


# ==============================
# PHASE 5 - EVALUATION
# ==============================

@app.route("/evaluation")
def evaluation():

    return render_template(
        "evaluation.html",
        active_page="crisp_dm"
    )


# ==============================
# PHASE 6 - DEPLOYMENT
# ==============================

@app.route("/deployment")
def deployment():

    return render_template(
        "deployment.html",
        active_page="crisp_dm"
    )

# ==============================
# EQUIPMENT REGISTRATION
# ==============================

@app.route("/application", methods=["GET", "POST"])
def application():

    success = None
    error = None

    if request.method == "POST":

        try:
            name = request.form.get("name", "").strip()

            equipment_type = request.form.get(
                "equipment_type", ""
            ).strip()

            brand = request.form.get("brand", "").strip()
            model = request.form.get("model", "").strip()

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

            if installation_date and installation_date > date.today():
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
                status="active"
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

    # Resolve alerts that no longer apply
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

            # Resolve duplicate active alerts
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
def dashboard():

    # Synchronize automatic maintenance alerts
    synchronize_all_automatic_alerts()

    # General system statistics
    total_equipment = Equipment.query.count()
    total_maintenance = MaintenanceRecord.query.count()
    total_predictions = Prediction.query.count()

    total_active_alerts = Alert.query.filter_by(
        status="active"
    ).count()

    # Prediction risk statistics
    low_risk = Prediction.query.filter_by(
        risk_level="Low"
    ).count()

    medium_risk = Prediction.query.filter_by(
        risk_level="Medium"
    ).count()

    high_risk = Prediction.query.filter_by(
        risk_level="High"
    ).count()

    # Latest five predictions
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
def maintenance():

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
def predictions():

    result = None
    error = None
    selected_equipment_id = None

    if request.method == "POST":

        try:
            # Get selected equipment
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

            # Get machine type
            machine_type = request.form.get(
                "machine_type",
                ""
            ).strip()

            if machine_type not in ["L", "M", "H"]:
                raise ValueError(
                    "Please select a valid machine type."
                )

            # Convert input values
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

            # Validate numeric values
            values = [
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear
            ]

            if not all(math.isfinite(value) for value in values):
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

            # Prepare data for the trained model
            equipment_data = {
                "Type": machine_type,
                "Air temperature [K]": air_temperature,
                "Process temperature [K]": process_temperature,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear
            }

            # Generate prediction
            result = predict_failure(equipment_data)

            # Save prediction in database
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

    # Retrieve registered equipment
    equipment_list = Equipment.query.order_by(
        Equipment.name.asc()
    ).all()

    # Retrieve prediction history
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
        selected_equipment_id=selected_equipment_id,
        active_page="predictions"
    )


# ==============================
# ALERT MANAGEMENT PAGE
# ==============================

@app.route("/alerts")
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
# CREATE DATABASE TABLES
# ==============================

with app.app_context():
    db.create_all()


# ==============================
# RUN APPLICATION
# ==============================

if __name__ == "__main__":
    app.run(debug=True)