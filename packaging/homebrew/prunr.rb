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
  desc "Remove backgrounds, erase, upscale and colorize photos on your own computer"
  homepage "https://prunr.io"
  version "0.5.1"
  license "Apache-2.0"

  depends_on macos: :sonoma

  on_macos do
    on_arm do
      url "https://github.com/aktiwers/prunr/releases/download/v#{version}/prunr-macos-aarch64.tar.gz"
      sha256 "03532cb2d31ecaba979b625cd98bb0a24708f322f468538109be019a5b1e30bf"
    end
  end

  def install
    # prunr loads ONNX Runtime from beside its real path, so both stay in
    # libexec and bin holds a link.
    libexec.install "prunr", "libonnxruntime.dylib"
    bin.install_symlink libexec/"prunr"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/prunr --version")
  end
end
