from datetime import datetime, timezone

from database import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(30), nullable=False, default="user")
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    equipment = db.relationship("Equipment", back_populates="owner")


class Equipment(db.Model):
    __tablename__ = "equipment"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    equipment_type = db.Column(db.String(100), nullable=False)
    brand = db.Column(db.String(100))
    model = db.Column(db.String(100))
    serial_number = db.Column(db.String(100), unique=True)
    location = db.Column(db.String(150))
    installation_date = db.Column(db.Date)
    last_maintenance_date = db.Column(db.Date)
    maintenance_interval_days = db.Column(db.Integer, nullable=False, default=90)
    status = db.Column(db.String(30), nullable=False, default="active")

    owner_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    owner = db.relationship("User", back_populates="equipment")
    maintenance_records = db.relationship(
        "MaintenanceRecord",
        back_populates="equipment",
        cascade="all, delete-orphan"
    )
    alerts = db.relationship(
        "Alert",
        back_populates="equipment",
        cascade="all, delete-orphan"
    )
    predictions = db.relationship(
        "Prediction",
        back_populates="equipment",
        cascade="all, delete-orphan"
    )


class MaintenanceRecord(db.Model):
    __tablename__ = "maintenance_records"

    id = db.Column(db.Integer, primary_key=True)
    equipment_id = db.Column(
        db.Integer,
        db.ForeignKey("equipment.id"),
        nullable=False
    )
    maintenance_date = db.Column(db.Date, nullable=False)
    maintenance_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)
    performed_by = db.Column(db.String(120))
    cost = db.Column(db.Float)
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    equipment = db.relationship(
        "Equipment",
        back_populates="maintenance_records"
    )


class Alert(db.Model):
    __tablename__ = "alerts"

    id = db.Column(db.Integer, primary_key=True)
    equipment_id = db.Column(
        db.Integer,
        db.ForeignKey("equipment.id"),
        nullable=False
    )
    alert_type = db.Column(db.String(50), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    is_read = db.Column(db.Boolean, default=False, nullable=False)
    status = db.Column(db.String(30), default="active", nullable=False)

    equipment = db.relationship("Equipment", back_populates="alerts")


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    equipment_id = db.Column(
        db.Integer,
        db.ForeignKey("equipment.id"),
        nullable=False
    )
    failure_probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(30), nullable=False)
    model_version = db.Column(db.String(50))
    predicted_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    equipment = db.relationship("Equipment", back_populates="predictions")