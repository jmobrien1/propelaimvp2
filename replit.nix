# replit.nix
{ pkgs }: {
  deps = [
    pkgs.python311Full
    pkgs.pkg-config
    pkgs.cairo
    pkgs.mesa
    pkgs.libglvnd
    pkgs.poppler_utils
    pkgs.tesseract # <-- The final required OCR engine
  ];
}