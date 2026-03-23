"""
api/routers/auth.py

Google OAuth2 flow + JWT auth via Authorization header.

Endpoints:
  GET  /auth/google           — Redirect to Google consent screen
  GET  /auth/google/callback  — Exchange code → redirect to frontend with token
  GET  /auth/me               — Return current user (reads Authorization header)
  POST /auth/logout           — No-op (frontend clears localStorage)

Dependency:
  get_current_user            — Reusable FastAPI dependency for protected routes
"""

from __future__ import annotations

import os
import secrets
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt

from api.models.user import User, get_or_create_user, get_user_by_id

router = APIRouter(prefix="/auth", tags=["auth"])

# ── Config ────────────────────────────────────────────────────────────────────

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
JWT_SECRET = os.environ.get("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 30

FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

# ── JWT helpers ───────────────────────────────────────────────────────────────


def _create_jwt(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _verify_jwt(token: str) -> str:
    """Decode JWT and return user_id, or raise 401."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: Optional[str] = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def _extract_token(request: Request) -> Optional[str]:
    """Extract JWT from Authorization header (Bearer <token>)."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


# ── Dependency ────────────────────────────────────────────────────────────────


async def get_optional_user(request: Request) -> Optional[User]:
    """Like get_current_user but returns None instead of raising 401."""
    token = _extract_token(request)
    if not token:
        return None
    try:
        user_id = _verify_jwt(token)
        return get_user_by_id(user_id)
    except HTTPException:
        return None


async def get_current_user(request: Request) -> User:
    """
    Reusable dependency. Reads JWT from the Authorization header.
    Raises 401 if missing or invalid.

    Usage:
        @router.get("/protected")
        async def protected(user: User = Depends(get_current_user)):
            ...
    """
    token = _extract_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user_id = _verify_jwt(token)
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/google")
async def google_login():
    """Redirect browser to Google's OAuth 2.0 consent screen."""
    state = secrets.token_urlsafe(16)
    redirect_uri = f"{BACKEND_URL}/auth/google/callback"
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    auth_url = f"{_GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"
    resp = RedirectResponse(url=auth_url)
    # Short-lived CSRF state cookie (same-origin, backend-only)
    resp.set_cookie(
        "oauth_state",
        state,
        max_age=300,
        httponly=True,
        samesite="lax",
        secure=BACKEND_URL.startswith("https"),
    )
    return resp


@router.get("/google/callback")
async def google_callback(
    code: str,
    state: str,
    oauth_state: Optional[str] = Cookie(None),
):
    """Exchange Google auth code for user info, redirect to frontend with token."""
    if not oauth_state or oauth_state != state:
        raise HTTPException(status_code=400, detail="OAuth state mismatch — possible CSRF")

    redirect_uri = f"{BACKEND_URL}/auth/google/callback"

    async with httpx.AsyncClient() as client:
        # Exchange code for access token
        token_resp = await client.post(
            _GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        if token_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to exchange authorization code")
        google_access_token = token_resp.json().get("access_token")

        # Fetch Google user profile
        info_resp = await client.get(
            _GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {google_access_token}"},
        )
        if info_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch Google user info")
        info = info_resp.json()

    user = get_or_create_user(
        google_id=info["id"],
        email=info["email"],
        name=info.get("name", ""),
        avatar_url=info.get("picture"),
    )
    jwt_token = _create_jwt(user.id)

    # Redirect to frontend callback page with token as query param
    dest = RedirectResponse(
        url=f"{FRONTEND_URL}/auth/callback?token={jwt_token}",
        status_code=302,
    )
    dest.delete_cookie("oauth_state")
    return dest


@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)):
    """Return the currently authenticated user."""
    return current_user


@router.post("/logout")
async def logout():
    """Logout is handled client-side by clearing localStorage."""
    return {"status": "logged out"}
