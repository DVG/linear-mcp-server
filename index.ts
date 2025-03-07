#!/usr/bin/env node

import { LinearClient, LinearDocument, Issue, User, Team, WorkflowState, IssueLabel } from "@linear/sdk";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequest,
  CallToolRequestSchema,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  ReadResourceRequestSchema,
  ListResourceTemplatesRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
  Tool,
  ResourceTemplate,
  Prompt,
} from "@modelcontextprotocol/sdk/types.js";
import dotenv from "dotenv";
import { z } from 'zod';

interface CreateIssueArgs {
  title: string;
  teamId: string;
  description?: string;
  priority?: number;
  status?: string;
}

interface UpdateIssueArgs {
  id: string;
  title?: string;
  description?: string;
  priority?: number;
  status?: string;
}

interface SearchIssuesArgs {
  query?: string;
  teamId?: string;
  limit?: number;
  status?: string;
  assigneeId?: string;
  labels?: string[];
  priority?: number;
  estimate?: number;
  includeArchived?: boolean;
}

interface GetUserIssuesArgs {
  userId?: string;
  includeArchived?: boolean;
  limit?: number;
}

interface AddCommentArgs {
  issueId: string;
  body: string;
  createAsUser?: string;
  displayIconUrl?: string;
}

interface CreateProjectArgs {
  name: string;
  teamId: string;
  description?: string;
  icon?: string;
  color?: string;
  state?: string;
}

interface UpdateProjectArgs {
  id: string;
  name?: string;
  description?: string;
  icon?: string;
  color?: string;
  state?: string;
}

interface ListProjectsArgs {
  teamId?: string;
  includeArchived?: boolean;
  limit?: number;
}

interface ProjectContentArgs {
  id: string;
  content: string;
}

interface RateLimiterMetrics {
  totalRequests: number;
  requestsInLastHour: number;
  averageRequestTime: number;
  queueLength: number;
  lastRequestTime: number;
}

interface LinearIssueResponse {
  identifier: string;
  title: string;
  priority: number | null;
  status: string | null;
  stateName?: string;
  url: string;
}

interface LinearProjectResponse {
  id: string;
  name: string;
  description: string | null;
  url: string;
  state: string | null;
  icon: string | null;
  color: string | null;
  teamName: string | null;
}

class RateLimiter {
  public readonly requestsPerHour = 1400;
  private queue: (() => Promise<any>)[] = [];
  private processing = false;
  private lastRequestTime = 0;
  private readonly minDelayMs = 3600000 / this.requestsPerHour;
  private requestTimes: number[] = [];
  private requestTimestamps: number[] = [];

  async enqueue<T>(fn: () => Promise<T>, operation?: string): Promise<T> {
    const startTime = Date.now();
    const queuePosition = this.queue.length;

    console.error(`[Linear API] Enqueueing request${operation ? ` for ${operation}` : ''} (Queue position: ${queuePosition})`);

    return new Promise((resolve, reject) => {
      this.queue.push(async () => {
        try {
          console.error(`[Linear API] Starting request${operation ? ` for ${operation}` : ''}`);
          const result = await fn();
          const endTime = Date.now();
          const duration = endTime - startTime;

          console.error(`[Linear API] Completed request${operation ? ` for ${operation}` : ''} (Duration: ${duration}ms)`);
          this.trackRequest(startTime, endTime, operation);
          resolve(result);
        } catch (error) {
          console.error(`[Linear API] Error in request${operation ? ` for ${operation}` : ''}: `, error);
          reject(error);
        }
      });
      this.processQueue();
    });
  }

  private async processQueue() {
    if (this.processing || this.queue.length === 0) return;
    this.processing = true;

    while (this.queue.length > 0) {
      const now = Date.now();
      const timeSinceLastRequest = now - this.lastRequestTime;

      const requestsInLastHour = this.requestTimestamps.filter(t => t > now - 3600000).length;
      if (requestsInLastHour >= this.requestsPerHour * 0.9 && timeSinceLastRequest < this.minDelayMs) {
        const waitTime = this.minDelayMs - timeSinceLastRequest;
        await new Promise(resolve => setTimeout(resolve, waitTime));
      }

      const fn = this.queue.shift();
      if (fn) {
        this.lastRequestTime = Date.now();
        await fn();
      }
    }

    this.processing = false;
  }

  async batch<T>(items: any[], batchSize: number, fn: (item: any) => Promise<T>, operation?: string): Promise<T[]> {
    const batches = [];
    for (let i = 0; i < items.length; i += batchSize) {
      const batch = items.slice(i, i + batchSize);
      batches.push(Promise.all(
        batch.map(item => this.enqueue(() => fn(item), operation))
      ));
    }

    const results = await Promise.all(batches);
    return results.flat();
  }

  private trackRequest(startTime: number, endTime: number, operation?: string) {
    const duration = endTime - startTime;
    this.requestTimes.push(duration);
    this.requestTimestamps.push(startTime);

    // Keep only last hour of requests
    const oneHourAgo = Date.now() - 3600000;
    this.requestTimestamps = this.requestTimestamps.filter(t => t > oneHourAgo);
    this.requestTimes = this.requestTimes.slice(-this.requestTimestamps.length);
  }

  getMetrics(): RateLimiterMetrics {
    const now = Date.now();
    const oneHourAgo = now - 3600000;
    const recentRequests = this.requestTimestamps.filter(t => t > oneHourAgo);

    return {
      totalRequests: this.requestTimestamps.length,
      requestsInLastHour: recentRequests.length,
      averageRequestTime: this.requestTimes.length > 0
        ? this.requestTimes.reduce((a, b) => a + b, 0) / this.requestTimes.length
        : 0,
      queueLength: this.queue.length,
      lastRequestTime: this.lastRequestTime
    };
  }
}

class LinearMCPClient {
  private client: LinearClient;
  public readonly rateLimiter: RateLimiter;

  constructor(apiKey: string) {
    if (!apiKey) throw new Error("LINEAR_API_KEY environment variable is required");
    this.client = new LinearClient({ apiKey });
    this.rateLimiter = new RateLimiter();
  }

  private async getIssueDetails(issue: Issue) {
    const [statePromise, assigneePromise, teamPromise] = [
      issue.state,
      issue.assignee,
      issue.team
    ];

    const [state, assignee, team] = await Promise.all([
      this.rateLimiter.enqueue(async () => statePromise ? await statePromise : null),
      this.rateLimiter.enqueue(async () => assigneePromise ? await assigneePromise : null),
      this.rateLimiter.enqueue(async () => teamPromise ? await teamPromise : null)
    ]);

    return {
      state,
      assignee,
      team
    };
  }

  private addMetricsToResponse(response: any) {
    const metrics = this.rateLimiter.getMetrics();
    return {
      ...response,
      metadata: {
        ...response.metadata,
        apiMetrics: {
          requestsInLastHour: metrics.requestsInLastHour,
          remainingRequests: this.rateLimiter.requestsPerHour - metrics.requestsInLastHour,
          averageRequestTime: `${Math.round(metrics.averageRequestTime)}ms`,
          queueLength: metrics.queueLength,
          lastRequestTime: new Date(metrics.lastRequestTime).toISOString()
        }
      }
    };
  }

  async listIssues() {
    const result = await this.rateLimiter.enqueue(
      () => this.client.issues({
        first: 50,
        orderBy: LinearDocument.PaginationOrderBy.UpdatedAt
      }),
      'listIssues'
    );

    const issuesWithDetails = await this.rateLimiter.batch(
      result.nodes,
      5,
      async (issue) => {
        const details = await this.getIssueDetails(issue);
        return {
          uri: `linear-issue:///${issue.id}`,
          mimeType: "application/json",
          name: issue.title,
          description: `Linear issue ${issue.identifier}: ${issue.title}`,
          metadata: {
            identifier: issue.identifier,
            priority: issue.priority,
            status: details.state ? await details.state.name : undefined,
            assignee: details.assignee ? await details.assignee.name : undefined,
            team: details.team ? await details.team.name : undefined,
          }
        };
      },
      'getIssueDetails'
    );

    return this.addMetricsToResponse(issuesWithDetails);
  }

  async getIssue(issueId: string) {
    const result = await this.rateLimiter.enqueue(() => this.client.issue(issueId));
    if (!result) throw new Error(`Issue ${issueId} not found`);

    const details = await this.getIssueDetails(result);

    return this.addMetricsToResponse({
      id: result.id,
      identifier: result.identifier,
      title: result.title,
      description: result.description,
      priority: result.priority,
      status: details.state?.name,
      assignee: details.assignee?.name,
      team: details.team?.name,
      url: result.url
    });
  }

  async createIssue(args: CreateIssueArgs) {
    const issuePayload = await this.client.createIssue({
      title: args.title,
      teamId: args.teamId,
      description: args.description,
      priority: args.priority,
      stateId: args.status
    });

    const issue = await issuePayload.issue;
    if (!issue) throw new Error("Failed to create issue");
    return issue;
  }

  async updateIssue(args: UpdateIssueArgs) {
    const issue = await this.client.issue(args.id);
    if (!issue) throw new Error(`Issue ${args.id} not found`);

    const updatePayload = await issue.update({
      title: args.title,
      description: args.description,
      priority: args.priority,
      stateId: args.status
    });

    const updatedIssue = await updatePayload.issue;
    if (!updatedIssue) throw new Error("Failed to update issue");
    return updatedIssue;
  }

  async searchIssues(args: SearchIssuesArgs) {
    const result = await this.rateLimiter.enqueue(() =>
      this.client.issues({
        filter: this.buildSearchFilter(args),
        first: args.limit || 10,
        includeArchived: args.includeArchived
      })
    );

    const issuesWithDetails = await this.rateLimiter.batch(result.nodes, 5, async (issue) => {
      const [state, assignee, labels] = await Promise.all([
        this.rateLimiter.enqueue(() => issue.state) as Promise<WorkflowState>,
        this.rateLimiter.enqueue(() => issue.assignee) as Promise<User>,
        this.rateLimiter.enqueue(() => issue.labels()) as Promise<{ nodes: IssueLabel[] }>
      ]);

      return {
        id: issue.id,
        identifier: issue.identifier,
        title: issue.title,
        description: issue.description,
        priority: issue.priority,
        estimate: issue.estimate,
        status: state?.name || null,
        assignee: assignee?.name || null,
        labels: labels?.nodes?.map((label: IssueLabel) => label.name) || [],
        url: issue.url
      };
    });

    return this.addMetricsToResponse(issuesWithDetails);
  }

  async getUserIssues(args: GetUserIssuesArgs) {
    try {
      const user = args.userId && typeof args.userId === 'string' ?
        await this.rateLimiter.enqueue(() => this.client.user(args.userId as string)) :
        await this.rateLimiter.enqueue(() => this.client.viewer);

      const result = await this.rateLimiter.enqueue(() => user.assignedIssues({
        first: args.limit || 50,
        includeArchived: args.includeArchived
      }));

      if (!result?.nodes) {
        return this.addMetricsToResponse([]);
      }

      const issuesWithDetails = await this.rateLimiter.batch(
        result.nodes,
        5,
        async (issue) => {
          const state = await this.rateLimiter.enqueue(() => issue.state) as WorkflowState;
          return {
            id: issue.id,
            identifier: issue.identifier,
            title: issue.title,
            description: issue.description,
            priority: issue.priority,
            stateName: state?.name || 'Unknown',
            url: issue.url
          };
        },
        'getUserIssues'
      );

      return this.addMetricsToResponse(issuesWithDetails);
    } catch (error) {
      console.error(`Error in getUserIssues: ${error}`);
      throw error;
    }
  }

  async addComment(args: AddCommentArgs) {
    const commentPayload = await this.client.createComment({
      issueId: args.issueId,
      body: args.body,
      createAsUser: args.createAsUser,
      displayIconUrl: args.displayIconUrl
    });

    const comment = await commentPayload.comment;
    if (!comment) throw new Error("Failed to create comment");

    const issue = await comment.issue;
    return {
      comment,
      issue
    };
  }

  async getTeamIssues(teamId: string) {
    const team = await this.rateLimiter.enqueue(() => this.client.team(teamId));
    if (!team) throw new Error(`Team ${teamId} not found`);

    const { nodes: issues } = await this.rateLimiter.enqueue(() => team.issues());

    const issuesWithDetails = await this.rateLimiter.batch(issues, 5, async (issue) => {
      const statePromise = issue.state;
      const assigneePromise = issue.assignee;

      const [state, assignee] = await Promise.all([
        this.rateLimiter.enqueue(async () => statePromise ? await statePromise : null),
        this.rateLimiter.enqueue(async () => assigneePromise ? await assigneePromise : null)
      ]);

      return {
        id: issue.id,
        identifier: issue.identifier,
        title: issue.title,
        description: issue.description,
        priority: issue.priority,
        status: state?.name,
        assignee: assignee?.name,
        url: issue.url
      };
    });

    return this.addMetricsToResponse(issuesWithDetails);
  }

  async getProjects(args: ListProjectsArgs = {}) {
    const filter: any = {};

    if (args.teamId) {
      filter.team = { id: { eq: args.teamId } };
    }

    if (args.includeArchived !== undefined) {
      filter.archived = { eq: args.includeArchived };
    }

    const result = await this.rateLimiter.enqueue(
      () => this.client.projects({
        first: args.limit || 50,
        filter
      }),
      'listProjects'
    );

    if (!result?.nodes) {
      return this.addMetricsToResponse([]);
    }

    const projectsWithDetails = await this.rateLimiter.batch(
      result.nodes,
      5,
      async (project) => {
        const team = await this.rateLimiter.enqueue(() => project.team);
        const state = await this.rateLimiter.enqueue(() => project.state);

        return {
          uri: `linear-project:///${project.id}`,
          mimeType: "application/json",
          name: project.name,
          description: `Linear project: ${project.name}`,
          metadata: {
            id: project.id,
            name: project.name,
            description: project.description,
            url: project.url,
            state: state?.name,
            icon: project.icon,
            color: project.color,
            teamName: team?.name
          }
        };
      },
      'getProjectDetails'
    );

    return this.addMetricsToResponse(projectsWithDetails);
  }

  async getProject(projectId: string) {
    const result = await this.rateLimiter.enqueue(() => this.client.project(projectId));
    if (!result) throw new Error(`Project ${projectId} not found`);

    const team = await this.rateLimiter.enqueue(() => result.team);
    const state = await this.rateLimiter.enqueue(() => result.state);

    return this.addMetricsToResponse({
      id: result.id,
      name: result.name,
      description: result.description,
      url: result.url,
      state: state?.name,
      icon: result.icon,
      color: result.color,
      teamName: team?.name
    });
  }

  async createProject(args: CreateProjectArgs) {
    const projectPayload = await this.client.createProject({
      name: args.name,
      teamId: args.teamId,
      description: args.description,
      icon: args.icon,
      color: args.color,
      stateId: args.state
    });

    const project = await projectPayload.project;
    if (!project) throw new Error("Failed to create project");
    return project;
  }

  async updateProject(args: UpdateProjectArgs) {
    const project = await this.client.project(args.id);
    if (!project) throw new Error(`Project ${args.id} not found`);

    const updatePayload = await project.update({
      name: args.name,
      description: args.description,
      icon: args.icon,
      color: args.color,
      stateId: args.state
    });

    const updatedProject = await updatePayload.project;
    if (!updatedProject) throw new Error("Failed to update project");
    return updatedProject;
  }

  async setProjectContent(args: ProjectContentArgs) {
    const project = await this.client.project(args.id);
    if (!project) throw new Error(`Project ${args.id} not found`);

    const contentPayload = await project.update({
      description: args.content
    });

    const updatedProject = await contentPayload.project;
    if (!updatedProject) throw new Error("Failed to update project content");
    return updatedProject;
  }

  async getViewer() {
    const viewer = await this.client.viewer;
    const [teams, organization] = await Promise.all([
      viewer.teams(),
      this.client.organization
    ]);

    return this.addMetricsToResponse({
      id: viewer.id,
      name: viewer.name,
      email: viewer.email,
      admin: viewer.admin,
      teams: teams.nodes.map(team => ({
        id: team.id,
        name: team.name,
        key: team.key
      })),
      organization: {
        id: organization.id,
        name: organization.name,
        urlKey: organization.urlKey
      }
    });
  }

  async getOrganization() {
    const organization = await this.client.organization;
    const [teams, users] = await Promise.all([
      organization.teams(),
      organization.users()
    ]);

    return this.addMetricsToResponse({
      id: organization.id,
      name: organization.name,
      urlKey: organization.urlKey,
      teams: teams.nodes.map(team => ({
        id: team.id,
        name: team.name,
        key: team.key
      })),
      users: users.nodes.map(user => ({
        id: user.id,
        name: user.name,
        email: user.email,
        admin: user.admin,
        active: user.active
      }))
    });
  }

  private buildSearchFilter(args: SearchIssuesArgs): any {
    const filter: any = {};

    if (args.query) {
      filter.or = [
        { title: { contains: args.query } },
        { description: { contains: args.query } }
      ];
    }

    if (args.teamId) {
      filter.team = { id: { eq: args.teamId } };
    }

    if (args.status) {
      filter.state = { name: { eq: args.status } };
    }

    if (args.assigneeId) {
      filter.assignee = { id: { eq: args.assigneeId } };
    }

    if (args.labels && args.labels.length > 0) {
      filter.labels = {
        some: {
          name: { in: args.labels }
        }
      };
    }

    if (args.priority) {
      filter.priority = { eq: args.priority };
    }

    if (args.estimate) {
      filter.estimate = { eq: args.estimate };
    }

    return filter;
  }
}

const createIssueTool: Tool = {
  name: "linear_create_issue",
  description: "Creates a new Linear issue with specified details. Use this to create tickets for tasks, bugs, or feature requests. Returns the created issue's identifier and URL. Required fields are title and teamId, with optional description, priority (0-4, where 0 is no priority and 1 is urgent), and status.",
  inputSchema: {
    type: "object",
    properties: {
      title: { type: "string", description: "Issue title" },
      teamId: { type: "string", description: "Team ID" },
      description: { type: "string", description: "Issue description" },
      priority: { type: "number", description: "Priority (0-4)" },
      status: { type: "string", description: "Issue status" }
    },
    required: ["title", "teamId"]
  }
};

const updateIssueTool: Tool = {
  name: "linear_update_issue",
  description: "Updates an existing Linear issue's properties. Use this to modify issue details like title, description, priority, or status. Requires the issue ID and accepts any combination of updatable fields. Returns the updated issue's identifier and URL.",
  inputSchema: {
    type: "object",
    properties: {
      id: { type: "string", description: "Issue ID" },
      title: { type: "string", description: "New title" },
      description: { type: "string", description: "New description" },
      priority: { type: "number", description: "New priority (0-4)" },
      status: { type: "string", description: "New status" }
    },
    required: ["id"]
  }
};

const searchIssuesTool: Tool = {
  name: "linear_search_issues",
  description: "Searches Linear issues using flexible criteria. Supports filtering by any combination of: title/description text, team, status, assignee, labels, priority (1=urgent, 2=high, 3=normal, 4=low), and estimate. Returns up to 10 issues by default (configurable via limit).",
  inputSchema: {
    type: "object",
    properties: {
      query: { type: "string", description: "Optional text to search in title and description" },
      teamId: { type: "string", description: "Filter by team ID" },
      status: { type: "string", description: "Filter by status name (e.g., 'In Progress', 'Done')" },
      assigneeId: { type: "string", description: "Filter by assignee's user ID" },
      labels: {
        type: "array",
        items: { type: "string" },
        description: "Filter by label names"
      },
      priority: {
        type: "number",
        description: "Filter by priority (1=urgent, 2=high, 3=normal, 4=low)"
      },
      estimate: {
        type: "number",
        description: "Filter by estimate points"
      },
      includeArchived: {
        type: "boolean",
        description: "Include archived issues in results (default: false)"
      },
      limit: {
        type: "number",
        description: "Max results to return (default: 10)"
      }
    }
  }
};

const getUserIssuesTool: Tool = {
  name: "linear_get_user_issues",
  description: "Retrieves issues assigned to a specific user or the authenticated user if no userId is provided. Returns issues sorted by last updated, including priority, status, and other metadata. Useful for finding a user's workload or tracking assigned tasks.",
  inputSchema: {
    type: "object",
    properties: {
      userId: { type: "string", description: "Optional user ID. If not provided, returns authenticated user's issues" },
      includeArchived: { type: "boolean", description: "Include archived issues in results" },
      limit: { type: "number", description: "Maximum number of issues to return (default: 50)" }
    }
  }
};

const addCommentTool: Tool = {
  name: "linear_add_comment",
  description: "Adds a comment to an existing Linear issue. Supports markdown formatting in the comment body. Can optionally specify a custom user name and avatar for the comment. Returns the created comment's details including its URL.",
  inputSchema: {
    type: "object",
    properties: {
      issueId: { type: "string", description: "ID of the issue to comment on" },
      body: { type: "string", description: "Comment text in markdown format" },
      createAsUser: { type: "string", description: "Optional custom username to show for the comment" },
      displayIconUrl: { type: "string", description: "Optional avatar URL for the comment" }
    },
    required: ["issueId", "body"]
  }
};

const createProjectTool: Tool = {
  name: "linear_create_project",
  description: "Creates a new Linear project with specified details. Use this to organize work in a project structure. Returns the created project's details including its URL. Required fields are name and teamId.",
  inputSchema: {
    type: "object",
    properties: {
      name: { type: "string", description: "Project name" },
      teamId: { type: "string", description: "Team ID" },
      description: { type: "string", description: "Project description in markdown format" },
      icon: { type: "string", description: "Project icon (emoji)" },
      color: { type: "string", description: "Project color (hex code)" },
      state: { type: "string", description: "Initial project state ID" }
    },
    required: ["name", "teamId"]
  }
};

const updateProjectTool: Tool = {
  name: "linear_update_project",
  description: "Updates an existing Linear project's properties. Use this to modify project details like name, description, icon, color, or state. Requires the project ID and accepts any combination of updatable fields. Returns the updated project details.",
  inputSchema: {
    type: "object",
    properties: {
      id: { type: "string", description: "Project ID" },
      name: { type: "string", description: "New project name" },
      description: { type: "string", description: "New project description in markdown format" },
      icon: { type: "string", description: "New project icon (emoji)" },
      color: { type: "string", description: "New project color (hex code)" },
      state: { type: "string", description: "New project state ID" }
    },
    required: ["id"]
  }
};

const listProjectsTool: Tool = {
  name: "linear_list_projects",
  description: "Lists Linear projects with optional filtering. Retrieves project details including name, description, state, team, and URLs. Useful for getting an overview of all projects or filtering by team.",
  inputSchema: {
    type: "object",
    properties: {
      teamId: { type: "string", description: "Filter projects by team ID" },
      includeArchived: { type: "boolean", description: "Include archived projects" },
      limit: { type: "number", description: "Maximum number of projects to return (default: 50)" }
    }
  }
};

const setProjectContentTool: Tool = {
  name: "linear_set_project_content",
  description: "Updates the description field of a Linear project with rich markdown content. Use this to provide detailed project documentation, specifications, or any other project-related content that supports markdown formatting.",
  inputSchema: {
    type: "object",
    properties: {
      id: { type: "string", description: "Project ID" },
      content: { type: "string", description: "Rich markdown content for the project description" }
    },
    required: ["id", "content"]
  }
};

const resourceTemplates: ResourceTemplate[] = [
  {
    uriTemplate: "linear-issue:///{issueId}",
    name: "Linear Issue",
    description: "A Linear issue with its details, comments, and metadata. Use this to fetch detailed information about a specific issue.",
    parameters: {
      issueId: {
        type: "string",
        description: "The unique identifier of the Linear issue (e.g., the internal ID)"
      }
    },
    examples: [
      "linear-issue:///c2b318fb-95d2-4a81-9539-f3268f34af87"
    ]
  },
  {
    uriTemplate: "linear-project:///{projectId}",
    name: "Linear Project",
    description: "A Linear project with its details, metadata, and rich content. Use this to fetch detailed information about a specific project.",
    parameters: {
      projectId: {
        type: "string",
        description: "The unique identifier of the Linear project (e.g., the internal ID)"
      }
    },
    examples: [
      "linear-project:///d3c429gc-06e3-5b92-0640-g4379g45bg98"
    ]
  },
  {
    uriTemplate: "linear-team:///{teamId}/issues",
    name: "Team Issues",
    description: "All active issues belonging to a specific Linear team, including their status, priority, and assignees.",
    parameters: {
      teamId: {
        type: "string",
        description: "The unique identifier of the Linear team (found in team settings)"
      }
    },
    examples: [
      "linear-team:///TEAM-123/issues"
    ]
  },
  {
    uriTemplate: "linear-viewer:",
    name: "Current User",
    description: "Information about the authenticated user associated with the API key, including their role, teams, and settings.",
    parameters: {},
    examples: [
      "linear-viewer:"
    ]
  },
  {
    uriTemplate: "linear-organization:",
    name: "Current Organization",
    description: "Details about the Linear organization associated with the API key, including settings, teams, and members.",
    parameters: {},
    examples: [
      "linear-organization:"
    ]
  },
  {
    uriTemplate: "linear-user:///{userId}/assigned",
    name: "User Assigned Issues",
    description: "Active issues assigned to a specific Linear user. Returns issues sorted by update date.",
    parameters: {
      userId: {
        type: "string",
        description: "The unique identifier of the Linear user. Use 'me' for the authenticated user"
      }
    },
    examples: [
      "linear-user:///USER-123/assigned",
      "linear-user:///me/assigned"
    ]
  }
];

const serverPrompt: Prompt = {
  name: "linear-server-prompt",
  description: "Instructions for using the Linear MCP server effectively",
  instructions: `This server provides access to Linear, a project management tool. Use it to manage issues, track work, coordinate with teams, and organize work in projects.

Key capabilities:
- Create and update issues: Create new tickets or modify existing ones with titles, descriptions, priorities, and team assignments.
- Search functionality: Find issues across the organization using flexible search queries with team and user filters.
- Team coordination: Access team-specific issues and manage work distribution within teams.
- Issue tracking: Add comments and track progress through status updates and assignments.
- Organization overview: View team structures and user assignments across the organization.
- Project management: Create, update, and organize work in projects with rich markdown content.

Tool Usage:
- linear_create_issue:
  - use teamId from linear-organization: resource
  - priority levels: 1=urgent, 2=high, 3=normal, 4=low
  - status must match exact Linear workflow state names (e.g., "In Progress", "Done")

- linear_update_issue:
  - get issue IDs from search_issues or linear-issue:/// resources
  - only include fields you want to change
  - status changes must use valid state IDs from the team's workflow

- linear_search_issues:
  - combine multiple filters for precise results
  - use labels array for multiple tag filtering
  - query searches both title and description
  - returns max 10 results by default

- linear_get_user_issues:
  - omit userId to get authenticated user's issues
  - useful for workload analysis and sprint planning
  - returns most recently updated issues first

- linear_add_comment:
  - supports full markdown formatting
  - use displayIconUrl for bot/integration avatars
  - createAsUser for custom comment attribution

- linear_create_project:
  - requires name and teamId
  - supports rich markdown descriptions
  - can set icon (emoji) and color (hex code)
  - state parameter accepts valid state IDs

- linear_update_project:
  - only include fields you want to change
  - use to update metadata like name, description, icon, color
  - state parameter accepts valid state IDs

- linear_list_projects:
  - filter by team to see team-specific projects
  - includeArchived parameter shows/hides archived projects
  - limit parameter controls number of results returned

- linear_set_project_content:
  - use to set rich markdown content for project descriptions
  - supports full markdown formatting including headers, lists, code blocks
  - useful for detailed project documentation

Best practices:
- When creating issues:
  - Write clear, actionable titles that describe the task well (e.g., "Implement user authentication for mobile app")
  - Include concise but appropriately detailed descriptions in markdown format with context and acceptance criteria
  - Set appropriate priority based on the context (1=critical to 4=nice-to-have)
  - Always specify the correct team ID (default to the user's team if possible)

- When searching:
  - Use specific, targeted queries for better results (e.g., "auth mobile app" rather than just "auth")
  - Apply relevant filters when asked or when you can infer the appropriate filters to narrow results

- When adding comments:
  - Use markdown formatting to improve readability and structure
  - Keep content focused on the specific issue and relevant updates
  - Include action items or next steps when appropriate

- When working with projects:
  - Use descriptive names that clearly identify the project's purpose
  - Provide detailed descriptions with markdown formatting for better readability
  - Use icons and colors to visually distinguish different projects
  - Organize related issues within projects for better work management
  - Use rich markdown content for comprehensive project documentation

- General best practices:
  - Fetch organization data first to get valid team IDs
  - Use search_issues to find issues for bulk operations
  - Include markdown formatting in descriptions and comments
  - Use projects to organize related work items

Resource patterns:
- linear-issue:///{issueId} - Single issue details (e.g., linear-issue:///c2b318fb-95d2-4a81-9539-f3268f34af87)
- linear-project:///{projectId} - Single project details (e.g., linear-project:///d3c429gc-06e3-5b92-0640-g4379g45bg98)
- linear-team:///{teamId}/issues - Team's issue list (e.g., linear-team:///OPS/issues)
- linear-user:///{userId}/assigned - User assignments (e.g., linear-user:///USER-123/assigned)
- linear-organization: - Organization for the current user
- linear-viewer: - Current user context

The server uses the authenticated user's permissions for all operations.`
};

interface MCPMetricsResponse {
  apiMetrics: {
    requestsInLastHour: number;
    remainingRequests: number;
    averageRequestTime: string;
    queueLength: number;
  }
}

// Zod schemas for tool argument validation
const CreateIssueArgsSchema = z.object({
  title: z.string().describe("Issue title"),
  teamId: z.string().describe("Team ID"),
  description: z.string().optional().describe("Issue description"),
  priority: z.number().min(0).max(4).optional().describe("Priority (0-4)"),
  status: z.string().optional().describe("Issue status")
});

const UpdateIssueArgsSchema = z.object({
  id: z.string().describe("Issue ID"),
  title: z.string().optional().describe("New title"),
  description: z.string().optional().describe("New description"),
  priority: z.number().optional().describe("New priority (0-4)"),
  status: z.string().optional().describe("New status")
});

const SearchIssuesArgsSchema = z.object({
  query: z.string().optional().describe("Optional text to search in title and description"),
  teamId: z.string().optional().describe("Filter by team ID"),
  status: z.string().optional().describe("Filter by status name (e.g., 'In Progress', 'Done')"),
  assigneeId: z.string().optional().describe("Filter by assignee's user ID"),
  labels: z.array(z.string()).optional().describe("Filter by label names"),
  priority: z.number().optional().describe("Filter by priority (1=urgent, 2=high, 3=normal, 4=low)"),
  estimate: z.number().optional().describe("Filter by estimate points"),
  includeArchived: z.boolean().optional().describe("Include archived issues in results (default: false)"),
  limit: z.number().optional().describe("Max results to return (default: 10)")
});

const GetUserIssuesArgsSchema = z.object({
  userId: z.string().optional().describe("Optional user ID. If not provided, returns authenticated user's issues"),
  includeArchived: z.boolean().optional().describe("Include archived issues in results"),
  limit: z.number().optional().describe("Maximum number of issues to return (default: 50)")
});

const AddCommentArgsSchema = z.object({
  issueId: z.string().describe("ID of the issue to comment on"),
  body: z.string().describe("Comment text in markdown format"),
  createAsUser: z.string().optional().describe("Optional custom username to show for the comment"),
  displayIconUrl: z.string().optional().describe("Optional avatar URL for the comment")
});

const CreateProjectArgsSchema = z.object({
  name: z.string().describe("Project name"),
  teamId: z.string().describe("Team ID"),
  description: z.string().optional().describe("Project description in markdown format"),
  icon: z.string().optional().describe("Project icon (emoji)"),
  color: z.string().optional().describe("Project color (hex code)"),
  state: z.string().optional().describe("Initial project state ID")
});

const UpdateProjectArgsSchema = z.object({
  id: z.string().describe("Project ID to update"),
  name: z.string().optional().describe("New project name"),
  description: z.string().optional().describe("New project description in markdown format"),
  icon: z.string().optional().describe("New project icon (emoji)"),
  color: z.string().optional().describe("New project color (hex code)"),
  state: z.string().optional().describe("New project state ID")
});

const ListProjectsArgsSchema = z.object({
  teamId: z.string().optional().describe("Filter projects by team ID"),
  includeArchived: z.boolean().optional().describe("Include archived projects"),
  limit: z.number().optional().describe("Maximum number of projects to return")
});

const ProjectContentArgsSchema = z.object({
  id: z.string().describe("Project ID"),
  content: z.string().describe("Rich markdown content for the project description")
});

async function main() {
  try {
    dotenv.config();

    const apiKey = process.env.LINEAR_API_KEY;
    if (!apiKey) {
      console.error("LINEAR_API_KEY environment variable is required");
      process.exit(1);
    }

    console.error("Starting Linear MCP Server...");
    const linearClient = new LinearMCPClient(apiKey);

    const server = new Server(
      {
        name: "linear-mcp-server",
        version: "1.0.0",
      },
      {
        capabilities: {
          prompts: {
            default: serverPrompt
          },
          resources: {
            templates: true,
            read: true
          },
          tools: {},
        },
      }
    );

    server.setRequestHandler(ListResourcesRequestSchema, async () => ({
      resources: await linearClient.listIssues()
    }));

    server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
      const uri = new URL(request.params.uri);
      const path = uri.pathname.replace(/^\//, '');

      try {
        if (uri.protocol === 'linear-organization:') {
          const organization = await linearClient.getOrganization();
          return {
            contents: [{
              uri: "linear-organization:",
              mimeType: "application/json",
              text: JSON.stringify(organization, null, 2)
            }]
          };
        }

        if (uri.protocol === 'linear-viewer:') {
          const viewer = await linearClient.getViewer();
          return {
            contents: [{
              uri: "linear-viewer:",
              mimeType: "application/json",
              text: JSON.stringify(viewer, null, 2)
            }]
          };
        }

        if (uri.protocol === 'linear-issue:') {
          const issue = await linearClient.getIssue(path);
          return {
            contents: [{
              uri: request.params.uri,
              mimeType: "application/json",
              text: JSON.stringify(issue, null, 2)
            }]
          };
        }

        if (uri.protocol === 'linear-project:') {
          const project = await linearClient.getProject(path);
          return {
            contents: [{
              uri: request.params.uri,
              mimeType: "application/json",
              text: JSON.stringify(project, null, 2)
            }]
          };
        }

        if (uri.protocol === 'linear-team:') {
          const [teamId] = path.split('/');
          const issues = await linearClient.getTeamIssues(teamId);
          return {
            contents: [{
              uri: request.params.uri,
              mimeType: "application/json",
              text: JSON.stringify(issues, null, 2)
            }]
          };
        }

        if (uri.protocol === 'linear-user:') {
          const [userId] = path.split('/');
          const issues = await linearClient.getUserIssues({
            userId
          });
          return {
            contents: [{
              uri: request.params.uri,
              mimeType: "application/json",
              text: JSON.stringify(issues, null, 2)
            }]
          };
        }

        throw new Error(`Unsupported resource URI: ${request.params.uri}`);
      } catch (error) {
        console.error("Error reading resource:", error);
        return {
          error: {
            type: 'UNKNOWN_ERROR',
            message: error instanceof Error ? error.message : String(error)
          }
        };
      }
    });

    server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [createIssueTool, updateIssueTool, searchIssuesTool, getUserIssuesTool, addCommentTool, createProjectTool, updateProjectTool, listProjectsTool, setProjectContentTool]
    }));

    server.setRequestHandler(ListResourceTemplatesRequestSchema, async () => {
      return {
        resourceTemplates: resourceTemplates
      };
    });

    server.setRequestHandler(ListPromptsRequestSchema, async () => {
      return {
        prompts: [serverPrompt]
      };
    });

    server.setRequestHandler(GetPromptRequestSchema, async (request) => {
      if (request.params.name === serverPrompt.name) {
        return {
          prompt: serverPrompt
        };
      }
      throw new Error(`Prompt not found: ${request.params.name}`);
    });

    server.setRequestHandler(CallToolRequestSchema, async (request: CallToolRequest) => {
      let metrics: RateLimiterMetrics = {
        totalRequests: 0,
        requestsInLastHour: 0,
        averageRequestTime: 0,
        queueLength: 0,
        lastRequestTime: Date.now()
      };

      try {
        const { name, arguments: args } = request.params;
        if (!args) throw new Error("Missing arguments");

        metrics = linearClient.rateLimiter.getMetrics();

        const baseResponse: MCPMetricsResponse = {
          apiMetrics: {
            requestsInLastHour: metrics.requestsInLastHour,
            remainingRequests: linearClient.rateLimiter.requestsPerHour - metrics.requestsInLastHour,
            averageRequestTime: `${Math.round(metrics.averageRequestTime)}ms`,
            queueLength: metrics.queueLength
          }
        };

        switch (name) {
          case "linear_create_issue": {
            const validatedArgs = CreateIssueArgsSchema.parse(args);
            const issue = await linearClient.createIssue(validatedArgs);
            return {
              content: [{
                type: "text",
                text: `Created issue ${issue.identifier}: ${issue.title}\nURL: ${issue.url}`,
                metadata: baseResponse
              }]
            };
          }

          case "linear_update_issue": {
            const validatedArgs = UpdateIssueArgsSchema.parse(args);
            const issue = await linearClient.updateIssue(validatedArgs);
            return {
              content: [{
                type: "text",
                text: `Updated issue ${issue.identifier}\nURL: ${issue.url}`,
                metadata: baseResponse
              }]
            };
          }

          case "linear_search_issues": {
            const validatedArgs = SearchIssuesArgsSchema.parse(args);
            const issues = await linearClient.searchIssues(validatedArgs);
            return {
              content: [{
                type: "text",
                text: `Found ${issues.length} issues:\n${issues.map((issue: LinearIssueResponse) =>
                  `- ${issue.identifier}: ${issue.title}\n  Priority: ${issue.priority || 'None'}\n  Status: ${issue.status || 'None'}\n  ${issue.url}`
                ).join('\n')
                  }`,
                metadata: baseResponse
              }]
            };
          }

          case "linear_get_user_issues": {
            const validatedArgs = GetUserIssuesArgsSchema.parse(args);
            const issues = await linearClient.getUserIssues(validatedArgs);

            return {
              content: [{
                type: "text",
                text: `Found ${issues.length} issues:\n${issues.map((issue: LinearIssueResponse) =>
                  `- ${issue.identifier}: ${issue.title}\n  Priority: ${issue.priority || 'None'}\n  Status: ${issue.status || 'None'}\n  ${issue.url}`
                ).join('\n')
                  }`,
                metadata: baseResponse
              }]
            };
          }

          case "linear_add_comment": {
            const validatedArgs = AddCommentArgsSchema.parse(args);
            const { comment, issue } = await linearClient.addComment(validatedArgs);

            return {
              content: [{
                type: "text",
                text: `Added comment to issue ${issue?.identifier}\nURL: ${comment.url}`,
                metadata: baseResponse
              }]
            };
          }

          case "linear_create_project": {
            const validatedArgs = CreateProjectArgsSchema.parse(args);
            const project = await linearClient.createProject(validatedArgs);
            return {
              content: [{
                type: "text",
                text: `Created project ${project.name}\nURL: ${project.url}`,
                metadata: baseResponse
              }]
            };
          }

          case "linear_update_project": {
            const validatedArgs = UpdateProjectArgsSchema.parse(args);
            const project = await linearClient.updateProject(validatedArgs);
            return {
              content: [{
                type: "text",
                text: `Updated project ${project.name}\nURL: ${project.url}`,
                metadata: baseResponse
              }]
            };
          }

          case "linear_list_projects": {
            const validatedArgs = ListProjectsArgsSchema.parse(args);
            const projects = await linearClient.getProjects(validatedArgs);
            return {
              content: [{
                type: "text",
                text: `Found ${projects.length} projects:\n${projects.map((project: LinearProjectResponse) =>
                  `- ${project.name}\n  Description: ${project.description || 'No description'}\n  ${project.url}`
                ).join('\n')
                  }`,
                metadata: baseResponse
              }]
            };
          }

          case "linear_set_project_content": {
            const validatedArgs = ProjectContentArgsSchema.parse(args);
            const project = await linearClient.setProjectContent(validatedArgs);
            return {
              content: [{
                type: "text",
                text: `Updated project ${project.name}\nURL: ${project.url}`,
                metadata: baseResponse
              }]
            };
          }

          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        console.error("Error executing tool:", error);

        const errorResponse: MCPMetricsResponse = {
          apiMetrics: {
            requestsInLastHour: metrics.requestsInLastHour,
            remainingRequests: linearClient.rateLimiter.requestsPerHour - metrics.requestsInLastHour,
            averageRequestTime: `${Math.round(metrics.averageRequestTime)}ms`,
            queueLength: metrics.queueLength
          }
        };

        // If it's a Zod error, format it nicely
        if (error instanceof z.ZodError) {
          const formattedErrors = error.errors.map(err => ({
            path: err.path,
            message: err.message,
            code: 'VALIDATION_ERROR'
          }));

          return {
            content: [{
              type: "text",
              text: {
                error: {
                  type: 'VALIDATION_ERROR',
                  message: 'Invalid request parameters',
                  details: formattedErrors
                }
              },
              metadata: {
                error: true,
                ...errorResponse
              }
            }]
          };
        }

        // For Linear API errors, try to extract useful information
        if (error instanceof Error && 'response' in error) {
          return {
            content: [{
              type: "text",
              text: {
                error: {
                  type: 'API_ERROR',
                  message: error.message,
                  details: {
                    // @ts-ignore - response property exists but isn't in type
                    status: error.response?.status,
                    // @ts-ignore - response property exists but isn't in type
                    data: error.response?.data
                  }
                }
              },
              metadata: {
                error: true,
                ...errorResponse
              }
            }]
          };
        }

        // For all other errors
        return {
          content: [{
            type: "text",
            text: {
              error: {
                type: 'UNKNOWN_ERROR',
                message: error instanceof Error ? error.message : String(error)
              }
            },
            metadata: {
              error: true,
              ...errorResponse
            }
          }]
        };
      }
    });

    const transport = new StdioServerTransport();
    console.error("Connecting server to transport...");
    await server.connect(transport);
    console.error("Linear MCP Server running on stdio");
  } catch (error) {
    console.error(`Fatal error in main(): ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  }
}

main().catch((error: unknown) => {
  console.error("Fatal error in main():", error instanceof Error ? error.message : String(error));
  process.exit(1);
});
