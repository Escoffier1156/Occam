{
  description = "Unified development environment";

  # Pinned to the exact nixpkgs revision this environment was already built from,
  # so locking costs no rebuild. Bump deliberately with `nix flake update`.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/31cd72fdba8fa052e437ce7e6879c4fe62def10f";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      # The package list stays in shell.nix; this flake only pins nixpkgs.
      devShells.${system}.default = import ./shell.nix { inherit pkgs; };
    };
}
