# This is the code I intend to use to create benchmark results and debugging values for TCCM
# It has three modes: (1) energy and gradient evaluation (with components) for the input
# structure; (2) geometry optimization with Cartesian coordinates, or (3) geometry optimization
# with redundant internal coordinates.

using Printf
using LinearAlgebra
using Zygote
# using Enzyme
# using ReverseDiff
# using AutoGrad

struct Shell
    atom::Int
    attyp::Int
    shtyp::Int
    aosh::Int
    nprsh::Int
    n0::Int
    zeta::Vector{Float64}
    djk::Vector{Float64}
    HA::Float64
    kCN::Float64
    kpoly::Float64
    EN::Float64
    rcov::Float64
    etasc::Float64
end

struct BFunc
    atom::Int
    attyp::Int
    ftyp::Int
    parsh::Int
end

struct Molec
    title::String
    labs::Vector{String}
    attyp::Vector{Int}
    nat::Int
    nelec::Int
    nshell::Int
    nbf::Int
    nocc::Int
    Zeff::Vector{Float64}
    alpha::Vector{Float64}
    rCovDisp::Vector{Float64}
    Gamma::Vector{Float64}
    shs::Vector{Shell}
    bfs::Vector{BFunc}
end

struct Params
    bohr_to_ang::Float64
    ang_to_bohr::Float64
    ha_to_ev::Float64
    nattyp::Int
    maxrefcn::Int
    atlabs::Vector{String}
    nrefCN::Vector{Int}
    refcn::Array{Float64,2}
    refc6::Array{Float64,4}
    Zeff::Vector{Float64}
    alpha::Vector{Float64}
    sqrtQA::Vector{Float64}
    RcovDisp::Vector{Float64}
    attypGamma::Vector{Float64}
    kllp::Array{Float64,2}
    kf::Float64
    a1::Float64
    a2::Float64
    s6::Float64
    s8::Float64
    kcn::Float64
    kl::Float64
    KABHH::Float64
    KABNH::Float64
    kEN::Float64
    bas::Array{Shell,2}
end
    
function ReadParams(filename::String)
   inputfile = open(filename)
   lines = readlines(inputfile)
   close(inputfile)
   auang::Float64 = parse(Float64,chomp(lines[7]))
   angau::Float64 = 1.0 / auang
   evau::Float64 = parse(Float64,chomp(lines[9]))
   nattyp::Int = parse(Int,chomp(lines[11]))
   atlabs::Vector{String} = split(lines[12],keepempty=false)
   kf::Float64 = parse(Float64,chomp(lines[16]))
   temp = split(lines[18],keepempty=false)
   alphas = zeros(nattyp)
   for i in 1:nattyp
      alphas[i] = parse(Float64,temp[i])
   end
   temp = split(lines[20],keepempty=false)
   zeff = zeros(nattyp)
   for i in 1:nattyp
      zeff[i] = parse(Float64,temp[i])
   end
   a1::Float64 = parse(Float64,split(lines[24],keepempty=false)[1])
   a2::Float64 = parse(Float64,split(lines[24],keepempty=false)[2])
   s6::Float64 = parse(Float64,split(lines[24],keepempty=false)[3])
   s8::Float64 = parse(Float64,split(lines[24],keepempty=false)[4])
   kcn::Float64 = parse(Float64,split(lines[24],keepempty=false)[5])
   kl::Float64 = parse(Float64,split(lines[24],keepempty=false)[6])
   temp = split(lines[26],keepempty=false)
   sqrtQA = zeros(nattyp)
   for i in 1:nattyp
      sqrtQA[i] = sqrt(parse(Float64,temp[i]))
   end
   temp = split(lines[28],keepempty=false)
   rcovdisp = zeros(nattyp)
   for i in 1:nattyp
      rcovdisp[i] = parse(Float64,temp[i]) * 4.0 / 3.0 * angau
   end
   maxrefcn::Int = parse(Int,chomp(lines[30]))
   temp = split(lines[32],keepempty=false)
   nrefcn = zeros(Int,maxrefcn)
   for i in 1:nattyp
      nrefcn[i] = parse(Int,temp[i])
   end
   refcn = zeros(Float64,maxrefcn,nattyp)
   for i in 1:nattyp
       k = 33+i
       temp = split(lines[k],keepempty=false)
       for j in 1:nrefcn[i]
          refcn[j,i] = parse(Float64,temp[j])
       end
   end
   npairs::Int = nattyp * (nattyp + 1) / 2
   refc6 = zeros(maxrefcn,maxrefcn,nattyp,nattyp)
   k = 35 + nattyp
   for i in 1:npairs
       k += 1
       atty1::Int = parse(Int,split(lines[k],keepempty=false)[1])
       atty2::Int = parse(Int,split(lines[k],keepempty=false)[2])
       for j in 1:nrefcn[atty1]
           k += 1
           temp = split(lines[k],keepempty=false)
           for jj in 1:nrefcn[atty2]
               refc6[j,jj,atty1,atty2] = parse(Float64,temp[jj])
               refc6[jj,j,atty2,atty1] = refc6[j,jj,atty1,atty2]
           end
       end
   end
   k += 7
   nzets = zeros(Int,2,nattyp)
   zets = zeros(8,2,nattyp)
   djks = zeros(8,2,nattyp)
   for i in 1:nattyp
       for j in 1:2
           k += 1
           ii = parse(Int,split(lines[k],keepempty=false)[1])
           jj = parse(Int,split(lines[k],keepempty=false)[2])
           kk = parse(Int,split(lines[k],keepempty=false)[3])
           nzets[j,i] = kk
           k += 1
           temp = split(lines[k],keepempty=false)
           for iii in 1:kk
               zets[iii,j,ii] = parse(Float64,temp[iii])
           end
           k += 1
           temp = split(lines[k],keepempty=false)
           for iii in 1:kk
               djks[iii,j,ii] = parse(Float64,temp[iii])
           end
       end
   end
   k += 4
   KABHH = parse(Float64,split(lines[k],keepempty=false)[1])
   KABNH = parse(Float64,split(lines[k],keepempty=false)[2])
   kllp::Array{Float64,2} = zeros(3,3)
   k += 2
   nkllp::Int = parse(Int,chomp(lines[k]))
   for i in 1:nkllp
      k += 1
      ii = parse(Int,split(lines[k],keepempty=false)[1])
      jj = parse(Int,split(lines[k],keepempty=false)[2])
      kllp[ii,jj] = parse(Float64,split(lines[k],keepempty=false)[3])
      if ii != jj
         kllp[jj,ii] = kllp[ii,jj]
      end
   end
   elecnegs = zeros(nattyp)
   k += 3
   temp = split(lines[k],keepempty=false)
   for i in 1:nattyp
      elecnegs[i] = parse(Float64,temp[i])
   end
   k += 2
   kEN::Float64 = parse(Float64,chomp(lines[k]))
   rcovpia = zeros(nattyp)
   k += 2
   temp = split(lines[k],keepempty=false)
   for i in 1:nattyp
      rcovpia[i] = parse(Float64,temp[i]) * angau
   end
   atGamma = zeros(nattyp)
   k += 3
   temp = split(lines[k],keepempty=false)
   for i in 1:nattyp
      atGamma[i] = parse(Float64,temp[i])
   end
   shn0 = zeros(Int,2,nattyp)
   shtyps = zeros(Int,2,nattyp)
   sheta = zeros(2,nattyp)
   shHA = zeros(2,nattyp)
   shkpoly = zeros(2,nattyp)
   k += 6
   for i in 1:nattyp
      k += 1
      temp = split(lines[k],keepempty=false)
      ii = parse(Int,temp[1])
      shtyps[1,ii] = parse(Int,temp[2]) 
      shn0[1,ii] = parse(Int,temp[3]) 
      sheta[1,ii] = parse(Float64,temp[4]) 
      shHA[1,ii] = parse(Float64,temp[5]) / evau
      shkpoly[1,ii] = parse(Float64,temp[6]) 
      k += 1
      temp = split(lines[k],keepempty=false)
      ii = parse(Int,temp[1])
      shtyps[2,ii] = parse(Int,temp[2]) 
      shn0[2,ii] = parse(Int,temp[3]) 
      sheta[2,ii] = parse(Float64,temp[4]) 
      shHA[2,ii] = parse(Float64,temp[5]) / evau
      shkpoly[2,ii] = parse(Float64,temp[6]) 
   end
   kcnvals = zeros(3)
   k += 2
   temp = split(lines[k],keepempty=false)
   for i in 1:3
      kcnvals[i] = parse(Float64,temp[i])
   end
   bas = Array{Shell}(undef,2,nattyp)
   for i in 1:nattyp
       for j in 1:2
           bas[j,i] = Shell(0,i,shtyps[j,i],0,nzets[j,i],shn0[j,i],zets[1:nzets[j,i],j,i],djks[1:nzets[j,i],j,i],
                      shHA[j,i], kcnvals[shtyps[j,i]], shkpoly[j,i],elecnegs[i], rcovpia[i],
                      sheta[j,i])
       end
   end
   return Params(auang,angau,evau,nattyp,maxrefcn,atlabs,nrefcn,refcn,refc6,zeff,alphas,sqrtQA,
                 rcovdisp,atGamma,kllp,kf,a1,a2,s6,s8,kcn,kl,KABHH,KABNH,kEN,bas)
end

function ReadXYZ(filename::String,bohr::Float64)
   inputfile = open(filename)
   lines = readlines(inputfile)
   close(inputfile)
   nat::Int = parse(Int,chomp(lines[1]))
   title::String = chomp(lines[2])
   x = zeros(Float64,3*nat)
   labs = Vector{String}(undef,nat)
   for i in 1:nat
       xloc = 3 * (i-1) + 1
       labs[i] = split(lines[i+2],keepempty=false)[1]
       x[xloc] = parse(Float64,split(lines[i+2],keepempty=false)[2])
       x[xloc+1] = parse(Float64,split(lines[i+2],keepempty=false)[3])
       x[xloc+2] = parse(Float64,split(lines[i+2],keepempty=false)[4])
   end
   x .*= bohr
   return labs, title, x
end

function BuildMolec(labs::Vector{String},title::String,opts::Params)
   nat::Int = size(labs)[1]
   bas = Vector{Shell}(undef,2*nat)
   atts = Vector{Int}(undef,nat)
   Zeff = Vector{Float64}(undef,nat)
   alpha = Vector{Float64}(undef,nat)
   rCovDisp = Vector{Float64}(undef,nat)
   Gamma = Vector{Float64}(undef,nat)
   kbas::Int = 0
   aosh::Int = 1
   nbf::Int = 0
   nelec::Int = 0
   for i in 1:nat
      attyp::Int = 0
      for k in 1:opts.nattyp
         if labs[i] == opts.atlabs[k]
             attyp = k
             break
         end
      end
      atts[i] = attyp
      Zeff[i] = opts.Zeff[atts[i]]
      alpha[i] = opts.alpha[atts[i]]
      rCovDisp[i] = opts.RcovDisp[atts[i]]
      Gamma[i] = opts.attypGamma[atts[i]]
      if attyp > 1
         nbf += 4
      elseif attyp == 1
         nbf += 2
      end
   end
   bfuncs = Vector{BFunc}(undef,nbf)
   kbf = 1
   for i in 1:nat
      for j in 1:2
         kbas += 1
         nelec += opts.bas[j,atts[i]].n0
         bas[kbas] = Shell(i,opts.bas[j,atts[i]].attyp,opts.bas[j,atts[i]].shtyp,aosh,
                          opts.bas[j,atts[i]].nprsh,opts.bas[j,atts[i]].n0,opts.bas[j,atts[i]].zeta,
                          opts.bas[j,atts[i]].djk,opts.bas[j,atts[i]].HA,opts.bas[j,atts[i]].kCN,
                          opts.bas[j,atts[i]].kpoly,opts.bas[j,atts[i]].EN,opts.bas[j,atts[i]].rcov,
                          opts.bas[j,atts[i]].etasc)
         if opts.bas[j,atts[i]].shtyp == 3
            aosh += 3
            bfuncs[kbf] = BFunc(i,atts[i],1,kbas) ; kbf += 1
            bfuncs[kbf] = BFunc(i,atts[i],2,kbas) ; kbf += 1
            bfuncs[kbf] = BFunc(i,atts[i],3,kbas) ; kbf += 1
         else
            aosh += 1
            bfuncs[kbf] = BFunc(i,atts[i],0,kbas)
            kbf += 1
         end
      end
   end
   return Molec(title,labs,atts,nat,nelec,2*nat,nbf,nelec/2,Zeff,alpha,rCovDisp,Gamma,bas,bfuncs)
end

function UpdateAtomShellCharges!(molec::Molec,qat::Vector{Float64},oldqat::Vector{Float64},
              qsh::Vector{Float64},oldqsh::Vector{Float64},pmat::Array{Float64,2},smat::Array{Float64,2})
sxp = [LinearAlgebra.dot(pmat[:,i],smat[:,i]) for i in 1:molec.nbf]
oldqsh .= qsh
j::Int = 1
for i in 1:molec.nshell
    if molec.shs[i].shtyp <= 2
        qsh[i] = molec.shs[i].n0 - sxp[j]
        j += 1
    elseif molec.shs[i].shtyp == 3
        qsh[i] = molec.shs[i].n0 - sum(sxp[j:j+2])
        j += 3
    end
end
oldqat .= qat
for i in 1:molec.nat
   qat[i] = qsh[2*i-1]+qsh[2*i]
end
end

function ShellCharge(molec::Molec,sxp::Vector{Float64},shi::Int)
    if molec.shs[shi].shtyp <= 2
        return  molec.shs[shi].n0 - sxp[molec.shs[shi].aosh]
    elseif molec.shs[shi].shtyp == 3
        return  molec.shs[shi].n0 - sum(sxp[molec.shs[shi].aosh:molec.shs[shi].aosh+2])
    end
end

function LagrangianClosure(molec::Molec,opts::Params,Cmat::Array{Float64,2},Pmat::Array{Float64,2},eorbs::Vector{Float64})
    return x::Vector{Float64} -> Lagrangian(x,molec,opts,Cmat,Pmat,eorbs)
end 

function Lagrangian(x::Vector{Float64},molec::Molec,opts::Params,Cmat::Array{Float64,2},Pmat::Array{Float64,2},eorbs::Vector{Float64})
    rAB2 = [Distance2(x,i,j) for i in 1:molec.nat, j in 1:molec.nat]
    rAB = sqrt.(rAB2)
    erep::Float64 = RepulsionE(x,molec,opts,rAB)
    CNs::Vector{Float64} = [CN(x,i,molec,opts,rAB) for i in 1:molec.nat]
    edisp::Float64 = DispersionE(molec,opts,CNs,rAB2)
    smat = [OverlapIntegral(x,molec,rAB2[molec.bfs[i].atom,molec.bfs[j].atom],i,j) for i in 1:molec.nbf, j in 1:molec.nbf]
    hAl = [molec.shs[i].HA * (1 + molec.shs[i].kCN * CNs[molec.shs[i].atom]) for i in 1:molec.nshell]
    h0 = [ZerothOrderHamiltonianElement(i,j,smat[i,j],rAB[molec.bfs[i].atom,molec.bfs[j].atom],hAl,molec,opts) for i in 1:molec.nbf, j in 1:molec.nbf]
    gmat = [sqrt(1 / (rAB2[molec.shs[i].atom,molec.shs[j].atom] + ((1/molec.shs[i].etasc + 1/molec.shs[j].etasc)/2)^2)) for i in 1:molec.nshell, j in 1:molec.nshell]
    sxp = [LinearAlgebra.dot(Pmat[:,i],smat[:,i]) for i in 1:molec.nbf]
    qsh = [ShellCharge(molec,sxp,i) for i in 1:molec.nshell]
    qat = [qsh[i]+qsh[i+1] for i in 1:2:molec.nshell]
    e1 = sum(Pmat .* h0)
    e2 = qsh' * gmat * qsh / 2
    e3 = sum(qat[i]^3 * molec.Gamma[i] for i in 1:molec.nat) / 3
    epsmat = Transpose(Cmat[:,1:molec.nocc]) * smat * Cmat[:,1:molec.nocc]
    eorthog = sum((epsmat[i,i] - 1) * eorbs[i] for i in 1:molec.nocc)
    println("Energies in Lagrangian:")
    println("erep, edisp, e1, e2, e3, eorthog:")
    println(erep," ", edisp," ", e1," ", e2," ", e3," ", eorthog)
    println("Total energy:")
    println(erep + edisp + e1 + e2 + e3 + eorthog)
    return erep + edisp + e1 + e2 + e3 - eorthog
end

function FockMatrix!(fmat::Array{Float64,2},qat::Vector{Float64},qsh::Vector{Float64},smat::Array{Float64,2},
              gmat::Array{Float64,2},molec::Molec)
shshift = [LinearAlgebra.dot(qsh,gmat[:,i]) + qat[molec.shs[i].atom]^2 * molec.Gamma[molec.shs[i].atom] for i in 1:molec.nshell]
for i in 1:molec.nshell
    sht1::Int = molec.shs[i].shtyp
    att1::Int = molec.shs[i].attyp
    aosh1::Int = molec.shs[i].aosh
    for j in i:molec.nshell
        sht2::Int = molec.shs[j].shtyp
        att2::Int = molec.shs[j].attyp
        aosh2::Int = molec.shs[j].aosh
        shiftterm = (shshift[i] + shshift[j]) / 2
        if (sht1 <= 2) && (sht2 <= 2)
            fmat[aosh1,aosh2] -= smat[aosh1,aosh2] * shiftterm
            fmat[aosh2,aosh1] = fmat[aosh1,aosh2]
        elseif (sht1 <= 2) && (sht2 == 3)
            fmat[aosh1,aosh2:aosh2+2] -= smat[aosh1,aosh2:aosh2+2] * shiftterm
            fmat[aosh2:aosh2+2,aosh1] = fmat[aosh1,aosh2:aosh2+2]
        elseif (sht1 == 3) && (sht2 <= 2)
            fmat[aosh1:aosh1+2,aosh2] -= smat[aosh1:aosh1+2,aosh2] * shiftterm
            fmat[aosh2,aosh1:aosh1+2] = fmat[aosh1:aosh1+2,aosh2]
        elseif (sht1 == 3) && (sht2 == 3)
            fmat[aosh1:aosh1+2,aosh2:aosh2+2] -= smat[aosh1:aosh1+2,aosh2:aosh2+2] * shiftterm
            fmat[aosh2:aosh2+2,aosh1:aosh1+2] = fmat[aosh1:aosh1+2,aosh2:aosh2+2]
        end
    end
end
end


function CN(x::Vector{Float64},ati::Int,molec::Molec,opts::Params,rAB::Array{Float64,2})
   return reduce(+,[1 / (1 + exp(-opts.kcn * ((molec.rCovDisp[ati] + molec.rCovDisp[j])/rAB[ati,j] - 1.0))) for j in 1:ati-1], init = 0.0) +
          reduce(+,[1 / (1 + exp(-opts.kcn * ((molec.rCovDisp[ati] + molec.rCovDisp[j])/rAB[ati,j] - 1.0))) for j in ati+1:molec.nat], init = 0.0)
end

function OverlapIntegral(x::Vector{Float64},molec::Molec,rAB2::Float64,bfa::Int,bfb::Int)
    function PrimitiveIntegral(pri::Int,prj::Int,shi::Int,shj::Int)
        zet = molec.shs[shi].zeta[pri] + molec.shs[shj].zeta[prj]
        xi = molec.shs[shi].zeta[pri] * molec.shs[shj].zeta[prj] / zet
        return molec.shs[shi].djk[pri] * molec.shs[shj].djk[prj] * exp(-xi * rAB2) * sqrt((pi/zet)^3)
    end
    function SPPrimitiveIntegral(pri::Int,prj::Int,shi::Int,shj::Int,lval::Int)
        xloci = 3 * (molec.shs[shi].atom - 1) ; xlocj = 3 * (molec.shs[shj].atom - 1)
        zet = molec.shs[shi].zeta[pri] + molec.shs[shj].zeta[prj]
        xi = molec.shs[shi].zeta[pri] * molec.shs[shj].zeta[prj] / zet
        xp = (molec.shs[shi].zeta[pri] * x[xloci+lval] + molec.shs[shj].zeta[prj] * x[xlocj+lval]) /  zet
        return molec.shs[shi].djk[pri] * molec.shs[shj].djk[prj] * exp(-xi * rAB2) * sqrt((pi/zet)^3) * (xp - x[xlocj+lval])
    end
    function PPPrimitiveIntegralNonDiag(pri::Int,prj::Int,shi::Int,shj::Int,lvali::Int,lvalj::Int)
        xloci = 3 * (molec.shs[shi].atom - 1) ; xlocj = 3 * (molec.shs[shj].atom - 1)
        zet = molec.shs[shi].zeta[pri] + molec.shs[shj].zeta[prj]
        xi = molec.shs[shi].zeta[pri] * molec.shs[shj].zeta[prj] / zet
        xpi = (molec.shs[shi].zeta[pri] * x[xloci+lvali] + molec.shs[shj].zeta[prj] * x[xlocj+lvali]) /  zet
        xpj = (molec.shs[shi].zeta[pri] * x[xloci+lvalj] + molec.shs[shj].zeta[prj] * x[xlocj+lvalj]) /  zet
        return molec.shs[shi].djk[pri] * molec.shs[shj].djk[prj] * exp(-xi * rAB2) * sqrt((pi/zet)^3) * (xpi - x[xloci+lvali]) * (xpj - x[xlocj+lvalj])
    end
    function PPPrimitiveIntegralDiag(pri::Int,prj::Int,shi::Int,shj::Int,lval)
        xloci = 3 * (molec.shs[shi].atom - 1) ; xlocj = 3 * (molec.shs[shj].atom - 1)
        zet = molec.shs[shi].zeta[pri] + molec.shs[shj].zeta[prj]
        xi = molec.shs[shi].zeta[pri] * molec.shs[shj].zeta[prj] / zet
        xp = (molec.shs[shi].zeta[pri] * x[xloci+lval] + molec.shs[shj].zeta[prj] * x[xlocj+lval]) /  zet
        fac = exp(-xi * rAB2) * sqrt((pi/zet)^3) * ((xp - x[xloci+lval])*(xp-x[xlocj+lval]) + 0.5 / zet)
        return molec.shs[shi].djk[pri] * molec.shs[shj].djk[prj] * fac
    end
    sh1 = molec.bfs[bfa].parsh ; sh2 = molec.bfs[bfb].parsh
    if (molec.bfs[bfa].ftyp == 0) && (molec.bfs[bfb].ftyp == 0)
        return sum([PrimitiveIntegral(i,j,sh1,sh2) for i in 1:molec.shs[sh1].nprsh for j in 1:molec.shs[sh2].nprsh])
    elseif (molec.bfs[bfa].ftyp == 0) && (molec.bfs[bfb].ftyp > 0)
        return sum([SPPrimitiveIntegral(i,j,sh1,sh2,molec.bfs[bfb].ftyp) for i in 1:molec.shs[sh1].nprsh for j in 1:molec.shs[sh2].nprsh])
    elseif (molec.bfs[bfa].ftyp > 0) && (molec.bfs[bfb].ftyp == 0)
        return sum([SPPrimitiveIntegral(j,i,sh2,sh1,molec.bfs[bfa].ftyp) for i in 1:molec.shs[sh1].nprsh for j in 1:molec.shs[sh2].nprsh])
    elseif (molec.bfs[bfa].ftyp > 0) && (molec.bfs[bfb].ftyp > 0) && (molec.bfs[bfa].ftyp != molec.bfs[bfb].ftyp)
        return sum([PPPrimitiveIntegralNonDiag(i,j,sh1,sh2,molec.bfs[bfa].ftyp,molec.bfs[bfb].ftyp) for i in 1:molec.shs[sh1].nprsh for j in 1:molec.shs[sh2].nprsh])
    elseif (molec.bfs[bfa].ftyp > 0) && (molec.bfs[bfb].ftyp > 0) && (molec.bfs[bfa].ftyp == molec.bfs[bfb].ftyp)
        return sum([PPPrimitiveIntegralDiag(i,j,sh1,sh2,molec.bfs[bfa].ftyp) for i in 1:molec.shs[sh1].nprsh for j in 1:molec.shs[sh2].nprsh])
    end
end

function SCFEnergy(x::Vector{Float64},molec::Molec,opts::Params)
    ethresh = 0.0000001
    rAB2 = [Distance2(x,i,j) for i in 1:molec.nat, j in 1:molec.nat]
    rAB = sqrt.(rAB2)
    erep::Float64 = RepulsionE(x,molec,opts,rAB)
    println("The repulsion energy is: ",erep)
    CNs::Vector{Float64} = [CN(x,i,molec,opts,rAB) for i in 1:molec.nat]
    println("Coord numbers: ",CNs)
    edisp::Float64 = DispersionE(molec,opts,CNs,rAB2)
    println("The dispersion energy is: ",edisp)
    smat = [OverlapIntegral(x,molec,rAB2[molec.bfs[i].atom,molec.bfs[j].atom],i,j) for i in 1:molec.nbf, j in 1:molec.nbf]
    println("The overlap matrix is:")
    for i in 1:molec.nbf
       for j in 1:molec.nbf
          @printf("%12.6f",smat[i,j])
       end 
       @printf("\n") 
    end 
    hAl = [molec.shs[i].HA * (1 + molec.shs[i].kCN * CNs[molec.shs[i].atom]) for i in 1:molec.nshell]
    h0 = [ZerothOrderHamiltonianElement(i,j,smat[i,j],rAB[molec.bfs[i].atom,molec.bfs[j].atom],hAl,molec,opts) for i in 1:molec.nbf, j in 1:molec.nbf]
    println("The zeroth-order matrix is:")
    for i in 1:molec.nbf
       for j in 1:molec.nbf
          @printf("%12.6f",h0[i,j])
       end     
       @printf("\n")
    end 
    gmat = [sqrt(1 / (rAB2[molec.shs[i].atom,molec.shs[j].atom] + ((1/molec.shs[i].etasc + 1/molec.shs[j].etasc)/2)^2)) for i in 1:molec.nshell, j in 1:molec.nshell]
    println("The gamma repulsion matrix is:")
    for i in 1:molec.nshell
       for j in 1:molec.nshell
          @printf("%12.6f",gmat[i,j])
       end
       @printf("\n")
    end
    Sev, Sevec = eigen(smat)
    Smin12::Array{Float64,2} = Sevec * Diagonal(1 ./ sqrt.(Sev)) * Sevec' 
    println("The inverse square root of the overlap matrix is:")
    for i in 1:molec.nbf
        for j in 1:molec.nbf
            @printf("%12.6f",Smin12[i,j])
        end
        @printf("\n")
    end
    Fprim::Array{Float64,2} = Smin12 * h0 * Smin12
    println("The zeroth-order Hamiltonian in the orthogonal basis is:")
    for i in 1:molec.nbf
        for j in 1:molec.nbf
            @printf("%12.6f",Fprim[i,j])
        end
        @printf("\n")
    end
    Fev, Fevec = eigen(Fprim)
    Cmat = Smin12 * Fevec
    println("The orbital eigenvalues from diagonalizing the zeroth-order Hamiltonian in the orthogonal basis is:")
    for j in 1:molec.nbf
        @printf("%12.6f",Fev[j])
    end
    @printf("\n")
    println("The initial set of eigenvectors are:")
    for i in 1:molec.nbf
        @printf("Orbital %4d:",i)
        for j in 1:molec.nbf
            @printf("%12.6f",Cmat[j,i])
        end
        @printf("\n")
    end
    Pmat = 2 * Cmat[:,1:molec.nocc] * Transpose(Cmat[:,1:molec.nocc])
    println("The initial density matrix is:")
    for i in 1:molec.nbf
        for j in 1:molec.nbf
            @printf("%12.6f",Pmat[j,i])
        end
        @printf("\n")
    end
    qsh::Vector{Float64} = zeros(molec.nshell)
    oldqsh::Vector{Float64} = zeros(molec.nshell)
    qat::Vector{Float64} = zeros(molec.nat)
    oldqat::Vector{Float64} = zeros(molec.nat)
    ehamil::Float64 = 0
    ehamilold::Float64 = 0
    maxscfcycles::Int = 50
    Fmat = zeros(molec.nbf,molec.nbf)
    for k in 1:maxscfcycles
       UpdateAtomShellCharges!(molec,qat,oldqat,qsh,oldqsh,Pmat,smat)
       if k == 1
           println("The initial set of atomic charges at first cycle:")
           for j in 1:molec.nat
               @printf("%8.4f",qat[j])
           end
           @printf("\n")
       end
       ehamilold = ehamil
       e1 = sum(Pmat .* h0)
       e2 = qsh' * gmat * qsh / 2
       e3 = sum(qat[i]^3 * molec.Gamma[i] for i in 1:molec.nat) / 3
       ehamil = e1 + e2 + e3
       @printf("Cycle %3d: E1, E2, E3, Eelec:%14.8f%14.8f%14.8f%14.8f\n",k,e1,e2,e3,ehamil)
       if abs(ehamil - ehamilold) < ethresh
           println("Convergence reached")
           etot = erep + edisp + ehamil
           return etot, [erep, edisp, e1, e2, e3, ehamil], [Cmat, Pmat, Fev]
       end
       DampCharges!(k,qat,oldqat,qsh,oldqsh)
       Fmat .= h0
       FockMatrix!(Fmat,qat,qsh,smat,gmat,molec)
       Fprim = Smin12 * Fmat * Smin12
       Fev, Fevec = eigen(Fprim)
       Cmat = Smin12 * Fevec
       Pmat = 2 * Cmat[:,1:molec.nocc] * Transpose(Cmat[:,1:molec.nocc])
    end
end
    
function ZerothOrderHamiltonianElement(bfa::Int,bfb::Int,selem::Float64,rAB::Float64,hAl::Vector{Float64},molec::Molec,opts::Params)
    if bfa == bfb
        return hAl[molec.bfs[bfa].parsh]
    elseif molec.bfs[bfa].atom == molec.bfs[bfb].atom
        return 0
    else
        sha::Int = molec.bfs[bfa].parsh ; shb = molec.bfs[bfb].parsh
        shta::Int = molec.shs[sha].shtyp ; shtb::Int = molec.shs[shb].shtyp
        atta::Int = molec.bfs[bfa].attyp ; attb::Int = molec.bfs[bfb].attyp
        if (atta == 1) && (attb == 1) && (shta == 1) && (shtb == 1)
            KAB = opts.KABHH
        elseif (atta == 1) && (attb == 3) && (shta == 1)
            KAB = opts.KABNH
        elseif (atta == 3) && (attb == 1) && (shtb == 1)
            KAB = opts.KABNH
        else
            KAB = 1.0
        end
        hABll = (hAl[sha] + hAl[shb]) / 2
        if (shta == 2) || (shtb == 2)
            enterm = 1.0
        else
            enterm = 1.0 + opts.kEN * (molec.shs[sha].EN - molec.shs[shb].EN)^2
        end
        rcovAB = molec.shs[sha].rcov + molec.shs[shb].rcov
        rrat = sqrt(rAB/rcovAB)
        pirab = (1.0 + molec.shs[sha].kpoly * rrat) * (1.0 + molec.shs[shb].kpoly * rrat)
        fac = KAB * opts.kllp[shta,shtb] * hABll * enterm * pirab
        return selem * fac
    end
end


function RepulsionE(x::Vector{Float64},molec::Molec,opts::Params,rAB::Array{Float64,2})
    return sum([molec.Zeff[i] * molec.Zeff[j] / rAB[i,j] * exp(-sqrt(molec.alpha[i]
              * molec.alpha[j]) * rAB[i,j]^opts.kf)] for i in 1:molec.nat for j in 1:i-1)[1]
end

function DispersionE(molec::Molec,opts::Params,CNs::Vector{Float64},rAB2::Array{Float64,2})
   lis = [exp(-opts.kl * (CNs[i] - opts.refcn[ii,molec.attyp[i]])^2) for ii in 1:opts.maxrefcn, i in 1:molec.nat]
   return sum([PairwiseDispersionE(i,j,opts,molec,lis[:,i],lis[:,j],
                   CNs[i],CNs[j],rAB2[i,j])] for i in 1:molec.nat-1 for j in i+1:molec.nat)[1]
end

function PairwiseDispersionE(ati::Int,atj::Int,opts::Params,molec::Molec,lii::Vector{Float64},lij::Vector{Float64},CNi::Float64,CNj::Float64,rij2::Float64)
    c8ovc6::Float64 = 3.0 * opts.sqrtQA[molec.attyp[ati]] * opts.sqrtQA[molec.attyp[atj]]
    rab0::Float64 = sqrt(c8ovc6)
    r0::Float64 = opts.a1 * rab0 + opts.a2
    f6 = rij2^3 / (rij2^3 + r0^6)
    f8 = rij2^4 / (rij2^4 + r0^8)
    sumlij = sum(lii[i] * lij[j] for i in 1:opts.nrefCN[molec.attyp[ati]], j in 1:opts.nrefCN[molec.attyp[atj]])
    sumc6lij = sum(opts.refc6[i,j,molec.attyp[ati],molec.attyp[atj]] * lii[i] * lij[j] for i in 1:opts.nrefCN[molec.attyp[ati]], j in 1:opts.nrefCN[molec.attyp[atj]])
    c6 = sumc6lij / sumlij
    return -opts.s6 * c6 / rij2^3 * f6 -opts.s8 * c8ovc6 * c6 / rij2^4 * f8
end

function DampCharges!(k::Int,qat::Vector{Float64},oldqat::Vector{Float64},qsh::Vector{Float64},oldqsh::Vector{Float64})
maxdq = maximum(abs.(qsh-oldqsh))
if (k<=3) || (maxdq > 0.001)
   qsh .= 0.6 .* oldqsh .+ 0.4 .* qsh
   qat .= 0.6 .* oldqat .+ 0.4 .* qat
end
end

function PrintBasisSet(opts::Params)
@printf("\nThe basis set in the parameters molecule has: %3d atom types.\n",opts.nattyp)
for i in 1:opts.nattyp
    for j in 1:2
        @printf("   Shell %2d on atom type %2d, of type %2d with %2d primitives.\n    Zetas:",j,i,
           opts.bas[j,i].shtyp, opts.bas[j,i].nprsh)
        for k in 1:opts.bas[j,i].nprsh
            @printf("%12.6f",opts.bas[j,i].zeta[k])
        end
        @printf("\n     djks:")
        for k in 1:opts.bas[j,i].nprsh
            @printf("%12.6f",opts.bas[j,i].djk[k])
        end
        @printf("\n")
    end
end
end

function PrintBasis(molec::Molec)
@printf("\nThe molecule has: %6d atoms, %6d shells and %6d basis functions.\n",molec.nat,molec.nshell,molec.nbf)
for i in 1:molec.nshell
    @printf("   Shell %4d on atom %4d, of type %2d with %2d primitives.\n    Zetas:",i,molec.shs[i].atom,
        molec.shs[i].shtyp,molec.shs[i].nprsh)
    for j in 1:molec.shs[i].nprsh
        @printf("%12.6f",molec.shs[i].zeta[j])
    end
    @printf("\n     djks:")
    for j in 1:molec.shs[i].nprsh
        @printf("%12.6f",molec.shs[i].djk[j])
    end
    @printf("\n")
end
end

function Distance2(x::Vector{Float64},i::Int,j::Int)
   return (x[3*i-2]-x[3*j-2])^2+(x[3*i-1]-x[3*j-1])^2+(x[3*i]-x[3*j])^2
end

function main(prelim::Int)
#    molec = ReadMol2(ARGS[1])
    opts = ReadParams("parameters3.dat")
    PrintBasisSet(opts)
    ats::Vector{String}, title::String, x::Vector{Float64} = ReadXYZ(ARGS[1],opts.ang_to_bohr)
    println("The parameters file has: ",opts.nattyp," atom types")
    println("The coordinates file has: ",size(ats)[1]," atoms")
    molec::Molec = BuildMolec(ats,title,opts)
    PrintBasis(molec)
    etot, eterms, wf = SCFEnergy(x,molec,opts)
    println("Final total energy: ",etot)
    println("Final energy components: repulsion energy, dispersion energy, E1, E2, E3, Ehamil")
    println(eterms)
    elag::Float64 = Lagrangian(x,molec,opts,wf[1],wf[2],wf[3])
    LagrangianForGradient = LagrangianClosure(molec,opts,wf[1],wf[2],wf[3])
    etot2::Float64 = LagrangianForGradient(x)
    println("The overall Lagrangian recomputed with the direct function is: ",etot2)
    grad=Zygote.gradient(LagrangianForGradient,x)[1]
#    f_tape = ReverseDiff.GradientTape(Lagrangian, (x,molec,opts,wf[1],wf[2],wf[3]))
    # g = AutoGrad.grad(LagrangianForGradient)
    # grad = g(x)
    # grad = Zygote.gradient(DirectEnergy,newx)[1]
    # grad=zeros(3*molec.nat)
    # autodiff(Reverse,LagrangianForGradient,Active,Duplicated(x,grad))
    # println("Total gradient from Enzyme:")
    # println("Total gradient from Zygote:")
    println("Total gradient from Zygote:")
    for i in 1:molec.nat
        j = 3 * i - 2
        @printf("%12.6f %12.6f %12.6f\n",grad[j],grad[j+1],grad[j+2])
    end   
end

main(1)

