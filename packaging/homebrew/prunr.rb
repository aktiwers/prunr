# Template for the Homebrew formula — the published version lives in a separate tap:
# https://github.com/aktiwers/homebrew-prunr
#
# Users install via: brew install aktiwers/prunr/prunr
#
# When releasing a new version:
#   1. Let CI publish the GitHub release (macOS tarball)
#   2. Compute sha256: curl -sL <url> | sha256sum
#   3. Update version + sha256 in the tap repo's Formula/prunr.rb

class Prunr < Formula
  desc "Local AI background removal — no cloud, no API keys"
  homepage "https://prunr.io"
  version "0.4.4"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/aktiwers/prunr/releases/download/v#{version}/prunr-macos-aarch64.tar.gz"
      sha256 "39904c12bbb9e1456cadb7ee40745f28f15fe0c73184c2cfdb07dc2dad1bf5a2"
    end
  end

  def install
    bin.install "prunr"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/prunr --version")
  end
end
