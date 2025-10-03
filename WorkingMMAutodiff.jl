# Based on the gradient obtained using automatic differentiation, I do GeomOpt in Cartesians

using Printf
using LinearAlgebra
using StaticArrays
using Zygote

const MVec3 = MVector{3,Float64}
const SVec3 = SVector{3,Float64}

mutable struct Atoms
    atno::Int
    xloc::Int
    lab::String
    Atoms() = new(0,0,"")
end

mutable struct Molec
    nat::Int
    nx::Int
    nbond::Int
    nangs::Int
    ntors::Int
    ats::Vector{Atoms}
    bonds::Array{Int,2}
    angs::Array{Int,2}
    tors::Array{Int,2}
    novdw::Array{Int,2}
end

struct Params
    vdw::Array{Float64,2}
    rch::Float64
    kch::Float64
    rcc::Float64
    kcc::Float64
    ahch::Float64
    khch::Float64
    ahcc::Float64
    khcc::Float64
    accc::Float64
    kccc::Float64
    ators::Float64
end
    
function ReadParams(filename::String)
   inputfile = open(filename)
   lines = readlines(inputfile)
   close(inputfile)
   epsh::Float64 = parse(Float64,split(lines[2],keepempty=false)[3])
   sigh::Float64 = parse(Float64,split(lines[2],keepempty=false)[2])
   epsc::Float64 = parse(Float64,split(lines[3],keepempty=false)[3])
   sigc::Float64 = parse(Float64,split(lines[3],keepempty=false)[2])
   epshh::Float64 = epsh
   sighh::Float64 = 2*sigh
   Bhh::Float64 = 4 * epshh * sighh^6
   Ahh::Float64 = Bhh * sighh^6
   gAhh::Float64 = 12 * Ahh ; gBhh = 6 * Bhh
   epscc::Float64 = epsc
   sigcc::Float64 = 2*sigc
   Bcc::Float64 = 4 * epscc * sigcc^6
   Acc::Float64 = Bcc * sigcc^6
   gAcc::Float64 = 12 * Acc ; gBcc = 6 * Bcc
   epshc::Float64 = sqrt(epsc * epsh)
   sighc::Float64 = 2*sqrt(sigh * sigc)
   Bhc::Float64 = 4 * epshc * sighc^6
   Ahc::Float64 = Bhc * sighc^6
   gAhc::Float64 = 12 * Ahc ; gBhc = 6 * Bhc
   rcc::Float64 = parse(Float64,split(lines[5],keepempty=false)[4])
   kcc::Float64 = parse(Float64,split(lines[5],keepempty=false)[3])
   rch::Float64 = parse(Float64,split(lines[6],keepempty=false)[4])
   kch::Float64 = parse(Float64,split(lines[6],keepempty=false)[3])
   ahch::Float64 = parse(Float64,split(lines[8],keepempty=false)[5]) / 180 * pi
   khch::Float64 = parse(Float64,split(lines[8],keepempty=false)[4])
   ahcc::Float64 = parse(Float64,split(lines[9],keepempty=false)[5]) / 180 * pi
   khcc::Float64 = parse(Float64,split(lines[9],keepempty=false)[4])
   accc::Float64 = parse(Float64,split(lines[10],keepempty=false)[5]) / 180 * pi
   kccc::Float64 = parse(Float64,split(lines[10],keepempty=false)[4])
   ators::Float64 = parse(Float64,chomp(lines[12]))
   return Params([Ahh Ahc Acc ; Bhh Bhc Bcc ; gAhh gAhc gAcc ; gBhh gBhc gBcc ], rch, kch, rcc, kcc, ahch, khch, ahcc, khcc, accc, kccc, ators)
end

function ReadMol2(filename::String)
   inputfile = open(filename)
   lines = readlines(inputfile)
   close(inputfile)
   nat::Int = parse(Int,split(lines[1],keepempty=false)[1])
   nx::Int = 3 * nat
   nbond::Int = parse(Int,split(lines[1],keepempty=false)[2])
   nC::Int = 0
   nCC::Int = 0
   ats = [Atoms() for i in 1:nat]
   x = zeros(Float64,nx)
   bonds=zeros(Int,2,nbond)
   connec = zeros(Int,nat,nat)
   catoms = zeros(Int,nat)
   j = 0
   for i in 1:nat
       pieces = split(lines[i+1],keepempty=false)
       ats[i].xloc = j+1
       x[j+1] = parse(Float64,pieces[1])
       x[j+2] = parse(Float64,pieces[2])
       x[j+3] = parse(Float64,pieces[3])
       j += 3
       ats[i].lab = pieces[4]
       if pieces[4] == "H"
          ats[i].atno = 1
       elseif pieces[4] == "C"
          ats[i].atno = 6
          nC += 1
          catoms[nC] = i
       end
   end
   ccbonds = zeros(Int,2,nbond)
   for i in 1:nbond
       pieces = split(lines[nat+1+i],keepempty=false)
       bonds[1,i] = parse(Int,pieces[1])
       bonds[2,i] = parse(Int,pieces[2])
       if (ats[bonds[1,i]].atno == 6) && (ats[bonds[2,i]].atno == 6)
          nCC += 1
          ccbonds[1,nCC] = bonds[1,i]
          ccbonds[2,nCC] = bonds[2,i]
       end
       connec[bonds[1,i],bonds[2,i]] = 1
       connec[bonds[2,i],bonds[1,i]] = 1
   end
   novdw = copy(connec)
   nangs = 6 * nC
   angs = zeros(Int,4,nangs)
   jang::Int = 0
   for i in 1:nC
       bats = findall(!iszero,connec[catoms[i],:])
       for j in 1:3
          for k in j+1:4
              jang += 1
              angs[1,jang] = bats[j]
              angs[2,jang] = catoms[i]
              angs[3,jang] = bats[k]
              if (ats[bats[j]].atno == 6) && (ats[bats[k]].atno == 6)
                  angs[4,jang] = 1
              elseif (ats[bats[j]].atno == 6) && (ats[bats[k]].atno == 1)
                  angs[4,jang] = 2
              elseif (ats[bats[j]].atno == 1) && (ats[bats[k]].atno == 6)
                  angs[4,jang] = 2
              elseif (ats[bats[j]].atno == 1) && (ats[bats[k]].atno == 1)
                  angs[4,jang] = 3
              end
              novdw[bats[j],bats[k]] = 1
              novdw[bats[k],bats[j]] = 1
          end
       end
   end
   ntors = 9 * nCC
   tors = zeros(Int,4,ntors)
   jang = 0
   for i in 1:nCC
       bats1 = findall(!iszero,connec[ccbonds[1,i],:])
       bats2 = findall(!iszero,connec[ccbonds[2,i],:])
       for j in 1:4
          if bats1[j] == ccbonds[2,i]
             continue
          end
          for k in 1:4
              if bats2[k] == ccbonds[1,i]
                 continue
              end
              jang += 1
              tors[1,jang] = bats1[j]
              tors[2,jang] = ccbonds[1,i]
              tors[3,jang] = ccbonds[2,i]
              tors[4,jang] = bats2[k]
          end
       end
   end
   return x, Molec(nat,nx,nbond,nangs,ntors,ats,bonds,angs,tors,novdw)
end


function GeomOpt(x::Vector{Float64},molec::Molec,opts::Params)
   alpha0::Float64 = 0.8
   gradtol::Float64 = 0.001
   maxcyc::Int = 1000
   nx::Int = molec.nx
   invHess = zeros(nx,nx)
   for i in 1:nx
      invHess[i,i] = 1.0 /300.0
   end
   ecurr = TotalEnergy(x,molec,opts)
   grad = zeros(nx)
   DirectEnergy = ClosureParams(molec,opts)
   grad=Zygote.gradient(DirectEnergy,x)[1]
   xold::Vector{Float64} = copy(x)
   gradold::Vector{Float64} = copy(grad)
   pk = zeros(nx)
   sk = zeros(nx)
   yk = zeros(nx)
   for i in 1:maxcyc
      alpha::Float64 = alpha0
      pk .= -invHess * grad
      rhsfac::Float64 = LinearAlgebra.dot(grad,pk)
      xold .= x
      for j in 1:20
         sk = alpha * pk
         Armijoc::Float64 = 0.1
         x = xold .+ sk
         etemp = TotalEnergy(x,molec,opts)
         if etemp > ecurr + Armijoc * alpha * rhsfac
             alpha *= 0.8
             x = copy(xold)
         else
             break
         end
      end
      enew = TotalEnergy(x,molec,opts)
      grad = Zygote.gradient(DirectEnergy,x)[1]
      yk = grad .- gradold
      sk = alpha * pk
      fac1::Float64 = LinearAlgebra.dot(sk,yk)
      fac2::Float64 = LinearAlgebra.dot(yk,invHess*yk)
      invHess .+= (fac1 + fac2) / (fac1^2) .* sk .* sk' .- ((invHess * yk) .* sk' + sk .* (yk' * invHess))/fac1
      grms::Float64 = sqrt(sum(grad .^2) / molec.nx)
      @printf("Cycle: %3d Old and new energies: %12.6f %12.6f And GRMS: %12.5f\n", i,ecurr,enew,grms)
      if grms < gradtol
          println("Optimization converged")
          break
      end
      gradold .= grad
      ecurr = enew
   end
end


function BendEnergy(x::Vector{Float64},molec::Molec,opts::Params)
   ebend::Float64 = 0.
   for i in 1:molec.nangs
      atA::Atoms = molec.ats[molec.angs[1,i]]
      atB::Atoms = molec.ats[molec.angs[2,i]]
      atC::Atoms = molec.ats[molec.angs[3,i]]
      angtyp::Int = molec.angs[4,i]
      vecBA::SVec3 = x[atA.xloc:atA.xloc+2] - x[atB.xloc:atB.xloc+2]
      vecBC::SVec3 = x[atC.xloc:atC.xloc+2] - x[atB.xloc:atB.xloc+2]
      angleABC::Float64 = BondAngle(vecBA,vecBC)
      if angtyp == 1
         ebend += opts.kccc * (angleABC - opts.accc)^2
      elseif angtyp == 2
         ebend += opts.khcc * (angleABC - opts.ahcc)^2
      elseif angtyp == 3
         ebend += opts.khcc * (angleABC - opts.ahch)^2
      end
   end
   return ebend
end

function TorsionEnergy(x::Vector{Float64},molec::Molec,opts::Params)
   etors::Float64 = 0
   for i in 1:molec.ntors
      atA::Atoms = molec.ats[molec.tors[1,i]]
      atB::Atoms = molec.ats[molec.tors[2,i]]
      atC::Atoms = molec.ats[molec.tors[3,i]]
      atD::Atoms = molec.ats[molec.tors[4,i]]
      vecAB::SVec3 = x[atB.xloc:atB.xloc+2] - x[atA.xloc:atA.xloc+2]
      vecBC::SVec3 = x[atC.xloc:atC.xloc+2] - x[atB.xloc:atB.xloc+2]
      vecCD::SVec3 = x[atD.xloc:atD.xloc+2] - x[atC.xloc:atC.xloc+2]
      vecT::SVec3 = VectorProduct(vecAB,vecBC)
      vecU::SVec3 = VectorProduct(vecBC,vecCD)
      vecTU::SVec3 = VectorProduct(vecT,vecU)
      rT2::Float64 = SquareLength(vecT)
      rU2::Float64 = SquareLength(vecU)
      vtvu::Float64 = sqrt(rT2 * rU2)
      rBC::Float64 = LinearAlgebra.norm(vecBC)
      cosangle::Float64 = LinearAlgebra.dot(vecT,vecU)/vtvu
      sinangle::Float64 = LinearAlgebra.dot(vecBC,vecTU)/(rBC * vtvu)
      cos3::Float64 = cosangle^3 - 3 * cosangle * sinangle^2
      etors += (1. + cos3)
   end
   return etors * opts.ators
end

function StretchEnergy(x::Vector{Float64},molec::Molec,opts::Params)
   estretch::Float64 = 0
   for i in 1:molec.nbond
      atA::Atoms = molec.ats[molec.bonds[1,i]]
      atB::Atoms = molec.ats[molec.bonds[2,i]]
      dist::Float64 = sqrt(SquareDistance(x,atA.xloc,atB.xloc))
      if atA.atno == 6 && atB.atno == 6
          estretch += opts.kcc * (dist - opts.rcc)^2
      else
          estretch += opts.kch * (dist - opts.rch)^2
      end
   end
   return estretch
end

function TotalEnergy(x::Vector{Float64},molec::Molec,opts::Params)
    return StretchEnergy(x,molec,opts) + BendEnergy(x,molec,opts) +
         TorsionEnergy(x,molec,opts) + VDWEnergy(x,molec,opts)
end

function ClosureParams(molec::Molec,opts::Params)
    return x::Vector{Float64} -> TotalEnergy(x,molec,opts)::Float64
end

function VDWEnergy(x::Vector{Float64},molec::Molec,opts::Params)
   evdw::Float64 = 0
   nvdw::Int = 0
   for i in 1:molec.nat
      atA::Atoms = molec.ats[i]
      for j in i+1:molec.nat
          if !iszero(molec.novdw[i,j])
             continue
          end
          atB::Atoms = molec.ats[j]
          r6::Float64 = (SquareDistance(x,atA.xloc,atB.xloc))^3
          nvdw += 1
          if (molec.ats[i].atno == 6) && (molec.ats[j].atno == 6)
             evdw += opts.vdw[1,3] / (r6^2) - opts.vdw[2,3] / (r6)
          elseif (molec.ats[i].atno == 1) && (molec.ats[j].atno == 1)
             evdw += opts.vdw[1,1] / (r6^2) - opts.vdw[2,1] / (r6) 
          else
             evdw += opts.vdw[1,2] / (r6^2) - opts.vdw[2,2] / (r6) 
          end 
      end
   end
   return evdw
end

function VectorProduct(v1::SVec3,v2::SVec3)
   return SVec3(v1[2] * v2[3] - v1[3] * v2[2], v1[3] * v2[1] - v1[1] * v2[3], v1[1] * v2[2] - v1[2] * v2[1])
end

function BondAngle(v1::SVec3,v2::SVec3)
   r1::Float64 = LinearAlgebra.norm(v1)
   r2::Float64 = LinearAlgebra.norm(v2)
   cosangle::Float64 = LinearAlgebra.dot(v1,v2)/(r1*r2)
   return acos(cosangle)
end

function SquareDistance(x::Vector{Float64},i::Int,j::Int)
   return ((x[i]-x[j])^2 + (x[i+1]-x[j+1])^2 + (x[i+2]-x[j+2])^2)
end

function SquareLength(v::SVec3)
   return (v[1]^2 + v[2]^2 + v[3]^2)
end

function PrintVec(lab::String,v::Vector{Float64})
    @printf("%s %13.6f %13.6f %13.6f\n",lab,v[1],v[2],v[3])
end  

function main(prelim::Int)
    x::Vector{Float64}, molec::Molec = ReadMol2("cyclohexane.mol2")
    opts::Params = ReadParams("tiny.parameters")
    println("The input file has: ",molec.nat," atoms")
    println("Atoms and coordinates:")
    for i in 1:molec.nat
        PrintVec(molec.ats[i].lab,x[molec.ats[i].xloc:molec.ats[i].xloc+2])
    end
    println("List of all bonds:")
    for i in 1:molec.nbond
       println(molec.bonds[1,i]," ",molec.bonds[2,i])
    end
    println("List of all atomic bends:")
    for i in 1:molec.nangs
       println(molec.angs[1,i]," ",molec.angs[2,i]," ",molec.angs[3,i]," ",molec.angs[4,i])
    end
    println("Bond stretching energy:")
    estretch::Float64 = StretchEnergy(x,molec,opts)
    @printf("%12.6f\n",estretch)
    println("Bending energy:")
    ebend::Float64 = BendEnergy(x,molec,opts)
    @printf("%12.6f\n",ebend)
    println("Torsional energy:")
    etors::Float64 = TorsionEnergy(x,molec,opts)
    @printf("%12.6f\n",etors)
    println("vdW energy:")
    evdw::Float64 = VDWEnergy(x,molec,opts)
    @printf("%12.6f\n",evdw)
    println("Total energy:")
    etot::Float64 = TotalEnergy(x,molec,opts)
    @printf("%12.6f\n",etot)
    DirectEnergy = ClosureParams(molec,opts)
    etot2::Float64 = DirectEnergy(x)
    println("Total energy with closure:")
    @printf("%12.6f\n",etot2)
    grad=Zygote.gradient(DirectEnergy,x)[1]
    println("Total gradient from Zygote:")
    for i in 1:molec.nat
        PrintVec(molec.ats[i].lab,grad[molec.ats[i].xloc:molec.ats[i].xloc+2])
    end
    nx = Val(molec.nx)
    @time GeomOpt(x,molec,opts)
end

main(1)

