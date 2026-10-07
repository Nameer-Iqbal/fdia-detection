# Environment verification script for Julia
using Pkg
using Random
using JuliaGrid
using CSV
using DataFrames

println("=== Julia Environment Check ===")
println("Julia Version: ", VERSION)
println("Active Project: ", Base.active_project())
println("\nPackage Status:")
Pkg.status()

println("\nLoaded Modules Verification:")
println(" - Random:     ", typeof(Random))
println(" - JuliaGrid:  ", typeof(JuliaGrid))
println(" - CSV:        ", typeof(CSV))
println(" - DataFrames: ", typeof(DataFrames))
println("\nAll Julia packages loaded successfully.")
