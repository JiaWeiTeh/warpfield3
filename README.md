# WARPFIELD3

<p><h2>A re-implementation of WARPFILEDv2.1</h2></p>

<p>This repo expands on WARPFIELD, and include changes such as:</p>
<ul>
    <li>The ability to include the effect of stochastic sampling of the IMF function in the formation of low-mass clusters</li>
    <li>Implement an easier-to-use file structure for future users and developers</li>
	<li>Create freedom to use different stellar evolution models (e.g. PARSEC, SB99)</li>
	<li>Improve flexibility in output format (ASCII, FITS, ...)</li>
	<li>Expand metallicity range</li>
</ul>
    
<p><h2>Repo Structure</h2></p>
<ul>
    <li><code>./lib</code> Directory containing libraries used by WARPFIELD.</li>
    <li><code>./param</code> Folder storing parameter files. Here, users can create a 
        <code>.param</code> file (following the syntax of <code>example.param</code>) to
        run WARPFIELD on.</li>
    <li><code>./src</code> Source code for WARPFIELD.
        <ul class="square">
          <li><code>/input_tools</code> useful tools to manipulate inputs</li>
          <li><code>/output_tools</code> useful tools to process outputs</li>
          <li><code>/warpfield</code> the main WARPFIELD code</li>
          <li><code>run.py</code>wrapper that takes in parameter file and runs WARPFIELD</li>
        </ul> 
    </li>
</ul> 
<p><h2>Running WARPFIELD</h2></p>
<p>To run, use the following in your terminal (an example):</p>
<code> python3 -m src.run param/example.param </code>
