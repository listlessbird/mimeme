import typer

from .scan import scan_command
from .verify import verify_command
from .reset_db import reset_db_command
from .upload import upload_command
from .backup_db import backup_db_command
from .restore_db import restore_db_command
from .rehydrate import rehydrate_command
from .annotate_cmd import annotate_command

app = typer.Typer()

app.command(name="scan")(scan_command)
app.command(name="verify")(verify_command)
app.command(name="reset-db")(reset_db_command)
app.command(name="upload")(upload_command)
app.command(name="backup-db")(backup_db_command)
app.command(name="restore-db")(restore_db_command)
app.command(name="rehydrate")(rehydrate_command)
app.command(name="annotate")(annotate_command)


def main():
    app()


if __name__ == "__main__":
    main()
