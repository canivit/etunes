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

      py =
        (pkgs.python312.withPackages (ps: [
          ps.numpy
          ps.pandas
          ps.scikit-learn
          ps.torch
          ps.torchvision
          (ps.opencv4.override { enableGtk2 = true; })
          ps.spotipy
          ps.python-dotenv
          ps.jupyter
          ps.tqdm
        ]));
    in
    {
      devShell = pkgs.mkShell {
        name = "py-flake";

        buildInputs = [
          py
        ];

        shellHook = ''
          mkdir -p .interpreter
          ln -sf ${py}/bin/python .interpreter/python
        '';
      };
    });
}
