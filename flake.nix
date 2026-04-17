{
  description = "A basic flake with a shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.systems.url = "github:nix-systems/default";
  inputs.flake-utils = {
    url = "github:numtide/flake-utils";
    inputs.systems.follows = "systems";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        deepspeed = pkgs.python3.pkgs.buildPythonPackage rec {
          pname = "deepspeed";
          version = "0.18.8";

          # format = "wheel";
          pyproject = true;

          src = pkgs.python3.pkgs.fetchPypi {
            inherit pname version;
            hash = "sha256-5OBRoUSwx0JwxG5JcBOfmoamH/JpWcXkYwAMSpO5kwQ=";
          };

          build-system = with pkgs.python3.pkgs; [
            setuptools
            wheel
          ];

          propagatedBuildInputs = with pkgs.python3.pkgs; [
            einops
            hjson
            msgpack
            ninja
            numpy
            packaging
            psutil
            py-cpuinfo
            pydantic
            torch
            tqdm
          ];

          # force CPU-only build
          preBuild = ''
            export DS_BUILD_OPS=0
            export DEEPSPEED_BUILD_OPS=0
          '';

          doCheck = false;

          enableParallelBuilding = true;
        };

      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (
              python-pkgs: with python-pkgs; [
                # select Python packages here
                deepspeed
                ftfy
                jinja2
                mpi4py
                numpy
                pybind11
                regex
                sentencepiece
                six
                tiktoken
                tokenizers
                transformers
              ]
            ))
          ];
        };
      }
    );
}
