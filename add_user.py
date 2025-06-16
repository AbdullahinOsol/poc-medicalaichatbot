# add_user.py
import bcrypt
from auth import Session, users

def add_user(username, raw_password):
    hashed = bcrypt.hashpw(raw_password.encode(), bcrypt.gensalt()).decode()
    session = Session()
    try:
        session.execute(users.insert().values(username=username, password_hash=hashed))
        session.commit()
        print("✅ User added successfully.")
    except Exception as e:
        session.rollback()
        print(f"❌ Failed to add user: {e}")
    finally:
        session.close()

# Replace with test credentials
add_user("new_user", "new_password")
