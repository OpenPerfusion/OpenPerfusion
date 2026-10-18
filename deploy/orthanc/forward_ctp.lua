-- Forward each CT series that looks like a perfusion acquisition (many images in one series) to the
-- OpenPerfusion node once the series is stable (no new instances for StableAge seconds). The node's own results
-- come back as OT (secondary capture) series and are not forwarded again.
function OnStableSeries(seriesId, tags, metadata)
  local series = ParseJson(RestApiGet('/series/' .. seriesId))
  local modality = series['MainDicomTags']['Modality']
  local n = #series['Instances']
  if modality == 'CT' and n >= 60 then
    print('OpenPerfusion: forwarding CT series ' .. seriesId .. ' (' .. n .. ' images)')
    RestApiPost('/modalities/openperfusion/store', seriesId)
  end
end
