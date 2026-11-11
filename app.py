from datetime import date, timedelta

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


app = Flask(__name__)

# ==============================
# DATABASE CONFIGURATION
# ==============================

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///predictive_maintenance.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# PGC PAGE
# ==============================

@app.route("/pgc")
def pgc():
    return render_template("pgc.html")


# ==============================
# CRISP-DM PAGE
# ==============================

@app.route("/crisp-dm")
def crisp_dm():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>CRISP-DM</title>
    </head>
    <body>
        <h1>CRISP-DM</h1>
        <p>This section is under development.</p>
        <a href="/">Return to Home</a>
    </body>
    </html>
    """)


# ==============================
# APPLICATION PAGE
# ==============================

@app.route("/application", methods=["GET", "POST"])
def application():

    success = None
    error = None

    if request.method == "POST":

        try:
            # Get form information
            name = request.form.get("name", "").strip()

            equipment_type = request.form.get(
                "equipment_type", ""
            ).strip()

            brand = request.form.get("brand", "").strip()
            model = request.form.get("model", "").strip()

            serial_number = request.form.get(
                "serial_number", ""
            ).strip()

            location = request.form.get("location", "").strip()

            interval_value = request.form.get(
                "maintenance_interval_days", "90"
            ).strip()

            installation_value = request.form.get(
                "installation_date", ""
            ).strip()

            # Validate required fields
            if not name or not equipment_type:
                raise ValueError(
                    "Equipment name and type are required."
                )

            interval = int(interval_value)

            if interval < 1:
                raise ValueError(
                    "Maintenance interval must be at least 1 day."
                )

            # Validate installation date
            installation_date = (
                date.fromisoformat(installation_value)
                if installation_value else None
            )

            # Check duplicate serial number
            if serial_number:
                existing_equipment = Equipment.query.filter_by(
                    serial_number=serial_number
                ).first()

                if existing_equipment:
                    raise ValueError(
                        "This serial number is already registered."
                    )

            # Create equipment record
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

            # Save to database
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
            app.logger.exception("Equipment registration failed")
            error = (
                "An unexpected error occurred. "
                "Please check the information and try again."
            )

    # Success message
    if request.args.get("registered") == "1":
        success = "Equipment registered successfully!"

    # Retrieve registered equipment
    equipment = Equipment.query.order_by(
        Equipment.id.desc()
    ).all()

    return render_template(
        "application.html",
        equipment=equipment,
        success=success,
        error=error
    )


# ==============================
# MAINTENANCE PAGE
# ==============================

@app.route("/maintenance", methods=["GET", "POST"])
def maintenance():

    success = None
    error = None

    # REGISTER MAINTENANCE
    if request.method == "POST":

        try:
            equipment_id = request.form.get(
                "equipment_id", type=int
            )

            maintenance_date = date.fromisoformat(
                request.form.get("maintenance_date", "")
            )

            maintenance_type = request.form.get(
                "maintenance_type", ""
            ).strip()

            description = request.form.get(
                "description", ""
            ).strip()

            performed_by = request.form.get(
                "performed_by", ""
            ).strip()

            cost_value = request.form.get(
                "cost", ""
            ).strip()

            # Validate equipment
            selected_equipment = db.session.get(
                Equipment, equipment_id
            )

            if selected_equipment is None:
                raise ValueError(
                    "Please select a valid equipment."
                )

            # Validate maintenance type
            if maintenance_type not in [
                "Preventive",
                "Corrective"
            ]:
                raise ValueError(
                    "Please select a valid maintenance type."
                )

            # Validate cost
            cost = float(cost_value) if cost_value else None

            if cost is not None and cost < 0:
                raise ValueError(
                    "Maintenance cost cannot be negative."
                )

            # Create maintenance record
            new_record = MaintenanceRecord(
                equipment_id=equipment_id,
                maintenance_date=maintenance_date,
                maintenance_type=maintenance_type,
                description=description,
                performed_by=performed_by,
                cost=cost
            )

            db.session.add(new_record)

            # Update last maintenance date
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

    # RETRIEVE EQUIPMENT
    equipment_list = Equipment.query.order_by(
        Equipment.name.asc()
    ).all()

    # RETRIEVE MAINTENANCE HISTORY
    records = MaintenanceRecord.query.order_by(
        MaintenanceRecord.maintenance_date.desc()
    ).all()

    # CALCULATE MAINTENANCE SCHEDULE
    maintenance_schedule = []

    today = date.today()
    alerts_changed = False

    # Alert types managed automatically by the system
    automatic_alert_types = [
        "maintenance_due_soon",
        "maintenance_overdue"
    ]

    for item in equipment_list:

        # Get last maintenance date or installation date
        reference_date = (
            item.last_maintenance_date
            or item.installation_date
        )

        if reference_date:

            next_maintenance = reference_date + timedelta(
                days=item.maintenance_interval_days
            )

            days_remaining = (
                next_maintenance - today
            ).days

            if days_remaining < 0:
                status = "Overdue"
                desired_alert_type = "maintenance_overdue"

            elif days_remaining <= 7:
                status = "Due Soon"
                desired_alert_type = "maintenance_due_soon"

            else:
                status = "Up to Date"
                desired_alert_type = None

        else:
            next_maintenance = None
            days_remaining = None
            status = "No Date"
            desired_alert_type = None

        # Add equipment schedule information
        maintenance_schedule.append({
            "equipment": item,
            "next_maintenance": next_maintenance,
            "days_remaining": days_remaining,
            "status": status
        })

        # Find active automatic alerts for this equipment
        active_alerts = Alert.query.filter(
            Alert.equipment_id == item.id,
            Alert.status == "active",
            Alert.alert_type.in_(automatic_alert_types)
        ).all()

        matching_alerts = [
            alert for alert in active_alerts
            if alert.alert_type == desired_alert_type
        ]

        # Resolve alerts that are no longer applicable
        for alert in active_alerts:

            if (
                desired_alert_type is None
                or alert.alert_type != desired_alert_type
            ):
                alert.status = "resolved"
                alerts_changed = True

        # Keep only one active alert of the required type
        if desired_alert_type and matching_alerts:

            for duplicate in matching_alerts[1:]:
                duplicate.status = "resolved"
                alerts_changed = True

        # Create a new alert if one does not already exist
        elif desired_alert_type:

            if desired_alert_type == "maintenance_overdue":
                message = (
                    f"Maintenance overdue for {item.name}. "
                    f"It was due on {next_maintenance}."
                )

            else:
                message = (
                    f"Maintenance due soon for {item.name}. "
                    f"Days remaining: {days_remaining}."
                )

            new_alert = Alert(
                equipment_id=item.id,
                alert_type=desired_alert_type,
                message=message,
                status="active",
                is_read=False
            )

            db.session.add(new_alert)
            alerts_changed = True

    # Save automatic alert changes
    if alerts_changed:
        db.session.commit()
        
    # SUCCESS MESSAGE
    if request.args.get("saved") == "1":
        success = (
            "Maintenance record registered successfully!"
        )

    # DISPLAY MAINTENANCE PAGE
    return render_template(
        "maintenance.html",
        equipment=equipment_list,
        records=records,
        maintenance_schedule=maintenance_schedule,
        success=success,
        error=error
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