export SpikeTrain, findspikes, addevent!, addcenter!, rate_bin, rate_KD, Kinematic

#=
Main data container of timestamps and trial data
=#

# inds - index of spikes associated with even
# time - timestamp of event
# id - type of event
# n  - occurs in nth trial
immutable event
    inds::UnitRange{Int64}
    time::Float64
    id::Int64
end
#    n::Int64
#end

# ts - timestamps of spikes
# trials - vector of events
# TD: Add dict to map event.id to a string
type SpikeTrain
    ts::Array{Float64,1}
    trials::Array{event,1}
end

# SpikeTrain Constructor
# spikes is an array of spike times
# times are event times
function SpikeTrain(spikes::Array{Float64,1},times)
    SpikeTrain(spikes,Array(event,length(times)))
end

function findspikes(spikes::Array{Array{Float64,1},1},times,eventIDs::Vector{Int},win::Float64)
    
    myspikes=Array(SpikeTrain,length(spikes))
    for i=1:length(spikes)
        myspikes[i]=SpikeTrain(spikes[i][:],times)
    end    
    
    addevent!(myspikes,times,win,eventIDs)

    myspikes 
end

# Just this one updated for now

# - times::Array{Array{Float64,1},1} -> unpack and add event ID vec for each inner array

# Quicker to do this way than to sort first
function findspikes(spikes::Array{Float64,2}, times::Vector{Float64}, eventIDs::Vector{Int},
    win::Float64) 
    spikeIDs = sub(spikes, :, 2);

    cells = sort(unique(spikeIDs))
    myspikes=Array(SpikeTrain,length(cells))
    count = 0
    for i in cells
        count+=1
        myspikes[count]=SpikeTrain(spikes[spikes[:,2].==i,1], times)
    end

    addevent!(myspikes, times, win, eventIDs)

    return myspikes
end

# Faster but assumes spikes sorted by CluID
function findsortedspikes(spikes::Array{Float64,2}, times::Vector{Float64}, eventIDs::Vector{Int},
    win::Float64)

    spikenums=sub(spikes,:,2)

    cells=unique(spikenums)
    cells=cells[2:end]
    myspikes=Array(SpikeTrain,length(cells))
    count=1
    firstind=1
    for i in cells
        lastind=searchsortedfirst(spikenums,i)
        myspikes[count]=SpikeTrain(spikes[firstind:(lastind-1),1],times)
        count+=1
        firstind=lastind
    end

    addevent!(myspikes,times,win,eventIDs)

    myspikes  
end

# Future - add a method for dict which generates IDs then saves in spikeTrain
# IDs implicit in order of ntimes array
function findspikes(spikes::Array{Float64,2}, ntimes::Array{Array{Float64,1}}, win::Float64)
    times, eventIDs = gettimeIDs(ntimes)
    findspikes(spikes, times, eventIDs, win)
end

function findsortedspikes(spikes::Array{Float64,2}, ntimes::Array{Array{Float64,1}}, win::Float64)
    times, eventIDs = gettimeIDs(ntimes)
    findsortedspikes(spikes, times, eventIDs, win)
end

function gettimeIDs(ntimes::Array{Array{Float64,1}})
    trigout = Array{Float64,2}(0,2)
    for (ind, trig) in enumerate(ntimes)
        trigin = cat(2, trig, repmat(collect(Float64(ind)), length(trig)))
        trigout = cat(1, trigout, trigin)
    end
    trigout = sortrows(trigout, by=x->x[1])
    return trigout[:,1], convert(Array{Int}, trigout[:,2])
end

#function findspikes(spikes::Array{Float64,2},times,win::Float64,cell_id)
#
#    spikenums=view(spikes,:,2)
#
#    cells=unique(spikenums)
#    cells=cells[2:end]
#    myspikes=Array(SpikeTrain,length(cells))
#    count=1
#    firstind=1
#    for i in cells
#        lastind=searchsortedfirst(spikenums,i)
#        myspikes[count]=SpikeTrain(spikes[firstind:(lastind-1),1],times)
#        count+=1
#        firstind=lastind
#    end
#
#    addevent!(myspikes,times,win)
#
#    (myspikes,[cell_id[round(Int64,i)] for i in cells])  
#end

function addevent!(spikes::Array{SpikeTrain,1},times::Vector{Float64},win::Float64, eventIDs::Vector{Int64})
    
    for i=1:length(spikes) # for each SpikeTrain
        first=searchsortedfirst(spikes[i].ts,times[1]-win)-1 # first spike for first time
        mysize=length(spikes[i].ts) # number of spikes
        if first<mysize
            for j=1:length(times)
                # get first and last spike within window
                first=searchsortedfirst(spikes[i].ts,(times[j]-win),first,mysize,Base.Forward)
                last=searchsortedfirst(spikes[i].ts, (times[j]+win),first,mysize,Base.Forward)-1
                spikes[i].trials[j]=Spikes.event(first:last,times[j],eventIDs[j]) # add event
            end
        end
    end
    nothing
end

#function addcenter!(spikes::Array{SpikeTrain,1},center::Array{Float64,1})
#      
#    for i=1:length(spikes)
#        spikes[i].center=hcat(spikes[i].center,center)
#    end
#    nothing
#end

#=
Rate Types
=#

abstract rate

#Bin
type rate_bin <: rate
    spikes::Array{SpikeTrain,1}
    binsize::Float64
end

#Kernel Density
type rate_KD <: rate
    spikes::Array{SpikeTrain,1}
    binsize::Float64
    kern::Float64

    function rate_KD(spikes::Array{SpikeTrain,1},binsize::Float64,kern::Float64)
        
        if kern<binsize
            kern=binsize
        end

        new(spikes,binsize,kern)
    end   
end

#=
Decoder Types
=#

export decoder, LDA, LeaveOne, Training

abstract classifier
abstract validation
abstract bias
abstract transformation

type decoder{C<:classifier,V<:validation} <: transformation
    classes::Array{Float64,1}
    c::C
    v::V
end

function decoder(c::classifier,v::validation)
    decoder(Array(Float64,0),c,v)
end

type LDA <: classifier
    myv::Array{Float64,1}
    W::Array{Float64,2}
    centroids::Array{Float64,2}
end

function LDA()
    LDA(Array(Float64,0),Array(Float64,0,0),Array(Float64,0,0))
end

type QDA <: classifier
end

type DLDA <: classifier
end

type DQDA <: classifier
end

type LeaveOne <: validation
end

type Training <: validation
    trainind::UnitRange{Int64}
    valind::UnitRange{Int64}
end

#=
Information Types
=#

export Information, QE

type Information{B<:bias,T<:transformation}
    b::B
    t::T
end

type QE <: bias
end

# Behavioral Types
# Position/Velocity x & y
type Kinematic
    px::Array{Float64,1}
    py::Array{Float64,1}
    vx::Array{Float64,1}
    vy::Array{Float64,1}
end

