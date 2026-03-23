from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class UserUpload(BaseModel):
    upload_id: str
    user_id: Optional[str] = None  # null for anonymous uploads (no FK target in users table)
    filename: str
    file_size_bytes: int
    bucket_path: str
    program_id: Optional[str] = None
    uploaded_at: datetime
    expires_at: datetime

    def to_db_row(self) -> dict:
        return {
            "upload_id": self.upload_id,
            "user_id": self.user_id,
            "filename": self.filename,
            "file_size_bytes": self.file_size_bytes,
            "bucket_path": self.bucket_path,
            "program_id": self.program_id,
            "uploaded_at": self.uploaded_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }

    @classmethod
    def from_db_row(cls, row: dict) -> "UserUpload":
        return cls(
            upload_id=row["upload_id"],
            user_id=row["user_id"],
            filename=row["filename"],
            file_size_bytes=row.get("file_size_bytes", 0),
            bucket_path=row["bucket_path"],
            program_id=row.get("program_id"),
            uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
        )
