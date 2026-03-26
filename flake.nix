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

          src = pkgs.python3.pkgs.fetchPypi {
            inherit pname version;
            hash = "sha256-5OBRoUSwx0JwxG5JcBOfmoamH/JpWcXkYwAMSpO5kwQ=";
          };

          # pyproject = true;

          build-system = with pkgs.python3.pkgs; [
            setuptools
            wheel
          ];

          nativeBuildInputs = with pkgs; [
            cmake
            ninja
          ];

          propagatedBuildInputs = with pkgs.python3.pkgs; [
            torch
            numpy
            psutil
            packaging
            py-cpuinfo
          ];

          DEEPSPEED_BUILD_OPS = "0";

          doCheck = false;
        };

      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (
              python-pkgs: with python-pkgs; [
                # select Python packages here
                matplotlib
                torch
                deepspeed
              ]
            ))
          ];
        };
      }
    );
}
