"""Report the registered conditional JSON-mode transport correction explicitly."""
import report_grounded_baselines_api as prior


if __name__ == '__main__':
    if (prior.BASE/'completion_receipt.json').exists():
        print('Completed results are frozen; no rewrite.')
    else:
        prior.main()
        path = prior.OUT/'report.txt'
        text = path.read_text(encoding='utf-8')
        text = text.replace('store=false, JSON-object output, no tools',
            'store=false, JSON-object transport only when the unchanged input contains the API-required json literal (otherwise default text output parsed as JSON), no tools')
        text += '\nFour initial LGBO API requests were rejected by input validation (HTTP400) because JSON-object mode requires the literal json in the input. No response or experimental choice was produced for those calls. Rejected transport records are retained; an independently reproduced minimal validation error and api_transport_registration_v2.json document the transport-only correction. Existing successful responses were not replaced.\n'
        path.write_text(text,encoding='utf-8')
