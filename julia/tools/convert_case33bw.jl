"""
julia/tools/convert_case33bw.jl

Reproduces data/cases/case33bw.m from data/cases/original/case33bw_original.m
Converts Baran & Wu 33-bus case into standard MATPOWER .m matrix format compatible with JuliaGrid.
"""

using Printf

function convert_case33bw(; input_path=nothing, output_path=nothing)
    root_dir = normpath(joinpath(@__DIR__, "..", ".."))
    if input_path === nothing
        input_path = joinpath(root_dir, "data", "cases", "original", "case33bw_original.m")
    end
    if output_path === nothing
        output_path = joinpath(root_dir, "data", "cases", "case33bw.m")
    end

    if !isfile(input_path)
        error("Input original case not found: $input_path")
    end

    lines = readlines(input_path)

    v_base = 12.66 * 1e3  # V
    s_base = 10.0 * 1e6   # VA
    z_base = (v_base^2) / s_base # 16.02756 Ohm

    clean_lines = String[]
    in_bus = false
    in_branch = false

    # Standard derived header
    header_inserted = false

    for line in lines
        # Stop at script execution section
        if startswith(strip(line), "[PQ, PV, REF") || startswith(strip(line), "%% convert")
            break
        end

        # Header adjustments
        if !header_inserted && startswith(line, "%% MATPOWER Case Format")
            push!(clean_lines, "%")
            push!(clean_lines, "%   Derived from MATPOWER's case33bw (3-Clause BSD License).")
            push!(clean_lines, "%   Source: https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case33bw.m")
            push!(clean_lines, "%   Conversions applied:")
            push!(clean_lines, "%     - Bus active/reactive loads converted from kW/kVAr to MW/MVAr (divided by 1000).")
            push!(clean_lines, "%     - Branch resistances/reactances converted from Ohms to p.u.")
            push!(clean_lines, "%       (Vbase = 12.66 kV, Sbase = 10 MVA, Zbase = 16.02756 Ohm).")
            push!(clean_lines, "%     - Stripped dynamic MATLAB script commands and opening-bracket comments")
            push!(clean_lines, "%       for compatibility with JuliaGrid's static matrix parser.")
            push!(clean_lines, "")
            header_inserted = true
        end

        if occursin("mpc.bus = [", line)
            in_bus = true
            push!(clean_lines, "mpc.bus = [")
            continue
        end
        if in_bus && occursin("];", line)
            in_bus = false
            push!(clean_lines, "];")
            continue
        end
        if occursin("mpc.branch = [", line)
            in_branch = true
            push!(clean_lines, "mpc.branch = [")
            continue
        end
        if in_branch && occursin("];", line)
            in_branch = false
            push!(clean_lines, "];")
            continue
        end

        if in_bus
            s = strip(line)
            if isempty(s) || startswith(s, "%")
                continue
            end
            parts = split(rstrip(s, ';'))
            if length(parts) >= 13
                bus_i = parts[1]
                b_type = parts[2]
                pd = parse(Float64, parts[3]) / 1000.0
                qd = parse(Float64, parts[4]) / 1000.0
                rest = join(parts[5:end], "\t")
                push!(clean_lines, string("\t", bus_i, "\t", b_type, "\t", @sprintf("%.6f", pd), "\t", @sprintf("%.6f", qd), "\t", rest, ";"))
            end
        elseif in_branch
            s = strip(line)
            if isempty(s) || startswith(s, "%")
                continue
            end
            parts = split(rstrip(s, ';'))
            if length(parts) >= 13
                fbus = parts[1]
                tbus = parts[2]
                r = parse(Float64, parts[3]) / z_base
                x = parse(Float64, parts[4]) / z_base
                rest = join(parts[5:end], "\t")
                push!(clean_lines, string("\t", fbus, "\t", tbus, "\t", @sprintf("%.8f", r), "\t", @sprintf("%.8f", x), "\t", rest, ";"))
            end
        else
            push!(clean_lines, line)
        end
    end

    open(output_path, "w") do f
        for l in clean_lines
            println(f, l)
        end
    end

    println("Successfully converted $input_path -> $output_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    convert_case33bw()
end
