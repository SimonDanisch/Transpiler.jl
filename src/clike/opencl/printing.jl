function Base.show(io::CLIO, x::Union{Float32, Float64})
    if isinf(x)
        print(io, "INFINITY")
    elseif isnan(x)
        print(io, "NAN")
    elseif isa(x, Float32)
        print(io, Float64(x), 'f')
    else
        print(io, Float64(x))
    end
end


function Sugar.gettypesource(x::CLMethod)
    T = x.signature
    tname = typename(EmptyCLIO(), T)
    str = if (!isleaftype(T) || T <: Type{X} where X) # emit type instances as singletons
        """typedef int $tname; // placeholder type instance
        __constant $tname TYP_INST_$tname;
        """
    else
        if isleaftype(T) && sizeof(T) == 0 && nfields(T) == 0
            # emit empty types as Int32, since struct can't be empty
            return "typedef int $tname; // empty type emitted as an int"
        else
            sprint() do io
                print(io, "struct  __attribute__ ((packed)) TYP$tname{\n")
                nf = nfields(T)
                fields = []
                if nf == 0 # structs can't be empty
                    # we use float as a short placeholder type.
                    # TODO, are there cases where float is no good?
                    println(io, "float empty; // structs can't be empty")
                else
                    for i in 1:nf
                        FT = fieldtype(T, i)
                        print(io, "    ", typename(EmptyCLIO(), FT))
                        print(io, ' ')
                        print(io, c_fieldname(T, i))
                        println(io, ';')
                    end
                end
                println(io, "};")
                println(io, "typedef struct TYP$tname $tname;")
            end
        end
    end
    return str
end


fixed_size_array_fieldname(::CLMethod, T, i) = "s$(i-1)"
