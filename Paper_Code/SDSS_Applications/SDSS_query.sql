SELECT
p.objid,p.ra,p.dec,p.u,p.g,p.r,p.i,p.z,
p.err_u, p.err_g, p.err_r, p.err_i, p.err_z,
p.run, p.rerun, p.camcol, p.field,
p.modelMag_u, p.modelMag_g, p.modelMag_r, p.modelMag_i, p.modelMag_z,
p.cModelMag_u, p.cModelMag_g, p.cModelMag_r, p.cModelMag_i, p.cModelMag_z,
p.extinction_u, p.extinction_g, p.extinction_r, p.extinction_i, p.extinction_z,
s.specobjid, s.plate as PLATE, s.mjd as MJD, s.fiberid as FIBERID, 
s.class, s.ra as Plug_RA, s.dec as Plug_dec, s.z as redshift,
s.snMedian_u, s.snMedian_g, s.snMedian_r, s.snMedian_i, s.snMedian_z, s.snMedian,
s.spectroFlux_u, s.spectroFlux_g, s.spectroFlux_r, s.spectroFlux_i, s.spectroFlux_z,
s.spectroFluxIvar_u, s.spectroFluxIvar_g, s.spectroFluxIvar_r, s.spectroFluxIvar_i, 
s.spectroFluxIvar_z, 
s.spectroSynFlux_u, s.spectroSynFlux_g, s.spectroSynFlux_r, s.spectroSynFlux_i, s.spectroSynFlux_z,
s.spectroSynFluxIvar_u, s.spectroSynFluxIvar_g, s.spectroSynFluxIvar_r, s.spectroSynFluxIvar_i, s.spectroSynFluxIvar_z,
s.spectroSkyFlux_u, s.spectroSkyFlux_g, s.spectroSkyFlux_r, s.spectroSkyFlux_i, s.spectroSkyFlux_z,
s.sn1_g, s.sn1_r, s.sn1_i, s.sn2_g, s.sn2_r, s.sn2_i
FROM PhotoObj AS p
JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE 
  s.z BETWEEN 0.38 AND 0.42
  AND s.class = 'GALAXY'
