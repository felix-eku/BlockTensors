function Base.join(f, io::IO, xs, delim, last)
    first = true
    local prev
    for x in xs
        if @isdefined prev
            first ? (first = false) : f(io, delim)
            f(io, prev)
        end
        prev = x
    end
    if @isdefined prev
        first || f(io, last)
        f(io, prev)
    end
    nothing
end
function Base.join(f, io::IO, xs, delim="")
    # Specialization of the above code when delim==last,
    # which lets us emit (compile) less code
    first = true
    for x in xs
        first ? (first = false) : f(io, delim)
        f(io, x)
    end
end