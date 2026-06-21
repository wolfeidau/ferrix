#!/usr/bin/env bash
set -euo pipefail

if ! command -v curl >/dev/null 2>&1 || ! command -v ssh-add >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends ca-certificates curl openssh-client
fi

sudo mkdir -p "$HOME/.local/bin" "$HOME/.local/share/mise" "$HOME/.cargo" "$HOME/.rustup" "$PWD/target"
sudo chown -R "$(id -u):$(id -g)" "$HOME/.local" "$HOME/.cargo" "$HOME/.rustup" "$PWD/target"

export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"

if ! command -v mise >/dev/null 2>&1; then
  curl https://mise.run | sh
fi

export PATH="$HOME/.local/bin:$HOME/.local/share/mise/shims:$HOME/.cargo/bin:$PATH"

if ! grep -q "Ferrix Rust tool homes" "$HOME/.bashrc"; then
  cat >>"$HOME/.bashrc" <<'EOF'

# Ferrix Rust tool homes.
export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"
export PATH="$HOME/.local/bin:$HOME/.local/share/mise/shims:$HOME/.cargo/bin:$PATH"
EOF
fi

if [ -f "$HOME/.zshrc" ] && ! grep -q "Ferrix Rust tool homes" "$HOME/.zshrc"; then
  cat >>"$HOME/.zshrc" <<'EOF'

# Ferrix Rust tool homes.
export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"
export PATH="$HOME/.local/bin:$HOME/.local/share/mise/shims:$HOME/.cargo/bin:$PATH"
EOF
fi

if ! grep -q "mise activate bash" "$HOME/.bashrc"; then
  cat >>"$HOME/.bashrc" <<'EOF'

# Activate mise-managed project tools.
if command -v mise >/dev/null 2>&1; then
  eval "$(mise activate bash)"
fi
EOF
fi

if [ -f "$HOME/.zshrc" ] && ! grep -q "mise activate zsh" "$HOME/.zshrc"; then
  cat >>"$HOME/.zshrc" <<'EOF'

# Activate mise-managed project tools.
if command -v mise >/dev/null 2>&1; then
  eval "$(mise activate zsh)"
fi
EOF
fi

mise trust "$PWD"
mise install
mise exec -- cargo fetch

install_buildkite_mcp_server() {
  local version="${BUILDKITE_MCP_SERVER_VERSION:-v1.6.1}"
  local arch
  case "$(uname -m)" in
    aarch64 | arm64) arch="arm64" ;;
    x86_64 | amd64) arch="x86_64" ;;
    *)
      echo "unsupported architecture for buildkite-mcp-server: $(uname -m)" >&2
      return 1
      ;;
  esac

  local tmpdir
  tmpdir="$(mktemp -d)"
  local archive="$tmpdir/buildkite-mcp-server.tar.gz"
  local url="https://github.com/buildkite/buildkite-mcp-server/releases/download/${version}/buildkite-mcp-server_Linux_${arch}.tar.gz"

  curl -fsSL "$url" -o "$archive"
  tar -xzf "$archive" -C "$tmpdir"
  install -m 0755 "$tmpdir/buildkite-mcp-server" "$HOME/.local/bin/buildkite-mcp-server"
  rm -rf "$tmpdir"
}

if ! command -v buildkite-mcp-server >/dev/null 2>&1; then
  install_buildkite_mcp_server
fi

git config --global core.sshCommand "ssh -o IdentityAgent=/agent.sock" || true
git config --global --unset-all gpg.ssh.program || true

mkdir -p "$HOME/.local/bin"
cat >"$HOME/.local/bin/git-ssh-keygen" <<'EOF'
#!/usr/bin/env bash
export SSH_AUTH_SOCK=/agent.sock
exec ssh-keygen "$@"
EOF
chmod +x "$HOME/.local/bin/git-ssh-keygen"
git config --global gpg.ssh.program "$HOME/.local/bin/git-ssh-keygen"