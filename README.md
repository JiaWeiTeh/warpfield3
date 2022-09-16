# WARPFIELD3

<p><h2>A re-implementation of WARPFILEDv2</h2></p>

<p>This repo expands on WARPFIELD2, and include changes such as:</p>
<ul>
    <li>The ability to include the effect of stochastic sampling of the IMF function in the formation of low-mass clusters.</li>
    <li>Implement an easier-to-use file structure for future users and developers.</li>
	<li>Create freedom to use different stellar evolution models (e.g. PARSEC, SB99).</li>
	<li>Improve flexibility in output format (ASCII, FITS, ...).</li>
	<li>Expand metallicity range.</li>
</ul>
    
<p><h2>Repo Structure</h2></p>
<ul>
    <li><code>./doc:</code> WARPFIELD documentation (TBD; check out the <a href="https://warpfield3.readthedocs.io/en/latest/" target="_blank" rel="noopener noreferrer">official page</a>).
    <li><code>./lib:</code> Directory containing libraries used by WARPFIELD.
    <ul class="square">
          <li><code>/cooling_tables:</code> OPIATE cooling tables. </li>
          <li><code>/imf:</code> Initial mass functions. </li>
          <li><code>/cloudy:</code> CLOUDY data. </li>
          <li><code>/sps:</code> SB99, SLUG, etc. </li>
        </ul></li>
    <li><code>./param:</code> Directory storing parameter files. Here, users can create a 
        <code>.param</code> file (following the syntax of <code>example.param</code>) to
        run WARPFIELD on.</li>
    <li><code>./src:</code> Directory containing source code for WARPFIELD.
        <ul class="square">
          <li><code>/input_tools:</code> Contains useful tools to manipulate inputs.</li>
          <li><code>/output_tools:</code> Contains useful tools to process outputs.</li>
          <li><code>/warpfield:</code> Contains the main WARPFIELD code.</li>
        </ul></li>
    <li><code>run.py:</code> A wrapper that takes in the parameter file and runs WARPFIELD.</li>
</ul> 
<p><h2>Running WARPFIELD</h2></p>
<p>To run, use the following in your terminal in the main WARPFIELD directory (an example):</p>
<code> python3 run.py param/example.param </code>