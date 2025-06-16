# auth.py
import bcrypt
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, select
from sqlalchemy.orm import sessionmaker

# Database connection string
DATABASE_URL = "sqlite:///mydatabase.db"

# Setup
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Users table definition
users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String, unique=True, nullable=False),
    Column("password_hash", String, nullable=False),
)

metadata.create_all(engine)

# Session maker
Session = sessionmaker(bind=engine)

# Credential verification function
def verify_credentials(username, password):
    with engine.connect() as conn:
        stmt = select(users).where(users.c.username == username)
        result = conn.execute(stmt).mappings().fetchone()
        if result:
            return bcrypt.checkpw(password.encode(), result["password_hash"].encode())
        return False
