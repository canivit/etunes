{
  description = "Nix flake based Python environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/master";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
      };
    in
    {
      devShell = pkgs.mkShell {
        name = "py-flake";

        buildInputs = with pkgs; [
          jetbrains.pycharm-community
          (python312.withPackages (ps: [
            ps.numpy
            ps.pandas
            ps.scikit-learn
            ps.torch
            ps.torchvision
            (ps.opencv4.override { enableGtk2 = true; })
          ]))
        ];
      };
    });
}
