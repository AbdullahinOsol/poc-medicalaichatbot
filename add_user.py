import argparse
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a new user to the chatbot system.")
    parser.add_argument("--username", required=True, help="Username for the new user")
    parser.add_argument("--password", required=True, help="Password for the new user")

    args = parser.parse_args()
    add_user(args.username, args.password)
