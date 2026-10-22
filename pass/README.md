# `pass/` — secrets and access guidance

This folder exists for the requested repository structure. **Do not store real passwords, API tokens, SSH keys, `.env` secrets, account recovery codes, dataset access tokens or user/customer data in Git.**

Use a password manager, local ignored `.env` files and platform secret stores (GitHub Actions Secrets, Streamlit secrets, or cloud secret managers). Keep `.env.example` non-secret and document only the expected environment-variable names. Verify secrets are absent before each commit.

If credentials are exposed, revoke/rotate immediately and coordinate repository/history and artifact cleanup. Deleting a secret in a newer Git commit does **not** make earlier copies inaccessible.

No secrets are required by the current local ReviewPulse inference pipeline. This file is documentation, not a credentials store.
