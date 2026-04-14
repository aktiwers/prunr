# Homebrew formula for prunr — local AI background removal
#
# Install: brew install aktiwers/prunr/prunr
# Requires a tap: brew tap aktiwers/prunr https://github.com/aktiwers/homebrew-prunr
#
# CI auto-updates version and sha256 on each release.

class Prunr < Formula
  desc "Local AI background removal — no cloud, no API keys"
  homepage "https://github.com/aktiwers/prunr"
  version "0.4.4"
  license any_of: ["MIT", "Apache-2.0"]

  on_macos do
    on_arm do
      url "https://github.com/aktiwers/prunr/releases/download/v#{version}/prunr-macos-aarch64.tar.gz"
      sha256 "PLACEHOLDER"
    end
  end

  def install
    bin.install "prunr"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/prunr --version")
  end
end
