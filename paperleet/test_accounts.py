"""Operator-only test account provisioning; no shared or default credentials."""
import argparse
import json
import sqlite3
import sys

from paperleet.accounts import AccountService, AccountError, AuthUnavailable
from paperleet.config import auth_mode, data_directory


def main(argv=None):
    parser = argparse.ArgumentParser(description="Provision individual PaperLeet beta test accounts")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("username")
    create.add_argument("--name", dest="display_name")
    create.add_argument("--admin", action="store_true", help="Grant this operator-created test account administrator access")
    commands.add_parser("reset-code").add_argument("username")
    commands.add_parser("disable").add_argument("username")
    commands.add_parser("list")
    args = parser.parse_args(argv)
    try:
        if auth_mode() != "test":
            raise AuthUnavailable("Set PAPERLEET_AUTH_MODE=test before provisioning test accounts")
        accounts = AccountService(data_directory() / "community.sqlite3")
        if args.command == "create":
            result = accounts.create_test_account(args.username, args.display_name, args.admin)
        elif args.command == "reset-code":
            result = accounts.reset_test_access_code(args.username)
        elif args.command == "disable":
            result = accounts.disable_test_account(args.username)
        else:
            result = accounts.list_test_accounts()
    except (AccountError, ValueError, OSError, sqlite3.Error) as exc:
        print("Test account error: " + str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
